import pickle  # nosec
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue

import grpc
import torch

from lerobot.async_inference.configs import DEFAULT_FPS, get_aggregate_function
from lerobot.async_inference.helpers import (
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
)
from lerobot.rl.acp_tags import build_acp_tagged_task
from lerobot.robots import Robot
from lerobot.scripts.recording_hil import ACPInferenceConfig
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks


@dataclass
class AsyncRemotePolicyConfig:
    policy_type: str
    pretrained_name_or_path: str
    actions_per_chunk: int
    server_address: str = "localhost:8080"
    policy_device: str = "cpu"
    client_device: str = "cpu"
    chunk_size_threshold: float = 0.5
    fps: int = DEFAULT_FPS
    aggregate_fn_name: str = "weighted_average"
    action_wait_timeout_s: float = 5.0
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.server_address:
            raise ValueError("server_address cannot be empty")
        if not self.policy_type:
            raise ValueError("policy_type cannot be empty")
        if not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path cannot be empty")
        if not self.policy_device:
            raise ValueError("policy_device cannot be empty")
        if not self.client_device:
            raise ValueError("client_device cannot be empty")
        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if not 0.0 <= self.chunk_size_threshold <= 1.0:
            raise ValueError(
                f"chunk_size_threshold must be between 0 and 1, got {self.chunk_size_threshold}"
            )
        if self.action_wait_timeout_s <= 0:
            raise ValueError(f"action_wait_timeout_s must be positive, got {self.action_wait_timeout_s}")
        self.aggregate_fn = get_aggregate_function(self.aggregate_fn_name)

    @property
    def environment_dt(self) -> float:
        return 1 / self.fps

    @property
    def type(self) -> str:
        return self.policy_type

    @property
    def pretrained_path(self) -> str:
        return self.pretrained_name_or_path

    @property
    def display_name(self) -> str:
        return Path(self.pretrained_name_or_path).name


@dataclass
class RemotePolicyRuntimeConfig:
    type: str
    pretrained_path: str
    device: str = "cpu"
    use_amp: bool = False


def resolve_async_inference_task(task: str | None, acp_inference: ACPInferenceConfig | None) -> str | None:
    if acp_inference is None or not acp_inference.enable:
        return task
    if acp_inference.use_cfg:
        raise ValueError("Async human-in-loop currently supports `acp_inference.enable=true` only with `use_cfg=false`.")
    return build_acp_tagged_task(task, is_positive=True)


class RemoteAsyncPolicyAdapter:
    logger = get_logger("async_hil_remote_policy")

    def __init__(self, cfg: AsyncRemotePolicyConfig, robot: Robot):
        self.adapter_config = cfg
        self.config = RemotePolicyRuntimeConfig(
            type=cfg.policy_type,
            pretrained_path=cfg.pretrained_name_or_path,
            device=cfg.client_device,
            use_amp=False,
        )
        self.server_address = cfg.server_address
        lerobot_features = map_robot_keys_to_lerobot_features(robot)
        self.policy_config = RemotePolicyConfig(
            cfg.policy_type,
            cfg.pretrained_name_or_path,
            lerobot_features,
            cfg.actions_per_chunk,
            cfg.policy_device,
            rename_map=cfg.rename_map,
        )
        self.channel = grpc.insecure_channel(
            self.server_address,
            grpc_channel_options(initial_backoff=f"{cfg.environment_dt:.4f}s"),
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.shutdown_event = threading.Event()
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1
        self.action_queue: Queue[TimedAction] = Queue()
        self.action_queue_lock = threading.Lock()
        self.must_go = threading.Event()
        self.must_go.set()
        self._receiver_thread: threading.Thread | None = None
        self._last_rpc_error: grpc.RpcError | None = None
        self._connected = False

    @property
    def running(self) -> bool:
        return self._connected and not self.shutdown_event.is_set()

    def start(self) -> None:
        self.stub.Ready(services_pb2.Empty())
        policy_config_bytes = pickle.dumps(self.policy_config)
        policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)
        self.stub.SendPolicyInstructions(policy_setup)
        self.shutdown_event.clear()
        self._connected = True
        self._receiver_thread = threading.Thread(target=self.receive_actions, daemon=True)
        self._receiver_thread.start()
        self.logger.info(
            "Async remote policy connected: server=%s policy=%s checkpoint=%s",
            self.server_address,
            self.adapter_config.policy_type,
            self.adapter_config.pretrained_name_or_path,
        )

    def stop(self) -> None:
        self.shutdown_event.set()
        if self._receiver_thread is not None:
            self._receiver_thread.join(timeout=1.0)
            self._receiver_thread = None
        self.channel.close()
        self._connected = False

    def reset(self) -> None:
        with self.action_queue_lock:
            self.action_queue = Queue()
        self.action_chunk_size = -1
        self.must_go.set()
        self._last_rpc_error = None

    def _ready_to_send_observation(self) -> bool:
        if self.action_chunk_size <= 0:
            return True
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
        return queue_size / self.action_chunk_size <= self.adapter_config.chunk_size_threshold

    def _actions_available(self) -> bool:
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _make_observation_payload(
        self,
        observation: RawObservation,
        task: str | None,
        acp_inference: ACPInferenceConfig | None,
    ) -> TimedObservation:
        payload = dict(observation)
        payload["task"] = resolve_async_inference_task(task, acp_inference)
        with self.latest_action_lock:
            timestep = max(self.latest_action, 0)
        with self.action_queue_lock:
            must_go = self.must_go.is_set() and self.action_queue.empty()
        return TimedObservation(
            timestamp=time.time(),
            observation=payload,
            timestep=timestep,
            must_go=must_go,
        )

    def send_observation(self, obs: TimedObservation) -> None:
        if not self.running:
            raise ConnectionError("Remote async policy is not running.")
        observation_bytes = pickle.dumps(obs)
        observation_iterator = send_bytes_in_chunks(
            observation_bytes,
            services_pb2.Observation,
            log_prefix="[ASYNC_HIL] Observation",
            silent=True,
        )
        try:
            self.stub.SendObservations(observation_iterator)
        except grpc.RpcError as error:
            self._last_rpc_error = error
            raise ConnectionError(f"Failed to send observation to remote policy server: {error}") from error

    def _aggregate_action_queues(self, incoming_actions: list[TimedAction]) -> None:
        future_action_queue: Queue[TimedAction] = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue
        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}
        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action
            if new_action.get_timestep() <= latest_action:
                continue
            if new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=self.adapter_config.aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )
        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self) -> None:
        while self.running:
            try:
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
            except grpc.RpcError as error:
                self._last_rpc_error = error
                time.sleep(0.05)
                continue
            if len(actions_chunk.data) == 0:
                continue
            timed_actions = pickle.loads(actions_chunk.data)  # nosec
            client_device = self.adapter_config.client_device
            if client_device != "cpu":
                for timed_action in timed_actions:
                    if timed_action.get_action().device.type != client_device:
                        timed_action.action = timed_action.get_action().to(client_device)
            self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))
            self._aggregate_action_queues(timed_actions)
            self.must_go.set()
            self._last_rpc_error = None

    def _pop_action(self) -> torch.Tensor:
        deadline = time.perf_counter() + self.adapter_config.action_wait_timeout_s
        while time.perf_counter() < deadline:
            with self.action_queue_lock:
                if not self.action_queue.empty():
                    timed_action = self.action_queue.get_nowait()
                    with self.latest_action_lock:
                        self.latest_action = timed_action.get_timestep()
                    return timed_action.get_action()
            if self._last_rpc_error is not None:
                raise ConnectionError(f"Failed to receive remote policy action: {self._last_rpc_error}")
            time.sleep(0.005)
        raise ConnectionError(
            f"Timed out waiting {self.adapter_config.action_wait_timeout_s:.2f}s for a remote policy action."
        )

    def predict_action_from_raw_observation(
        self,
        observation: RawObservation,
        task: str | None = None,
        robot_type: str | None = None,
        acp_inference: ACPInferenceConfig | None = None,
    ) -> torch.Tensor:
        del robot_type
        if not self.running:
            self.start()
        if self._ready_to_send_observation() or not self._actions_available():
            timed_observation = self._make_observation_payload(observation, task, acp_inference)
            self.send_observation(timed_observation)
            if timed_observation.must_go:
                self.must_go.clear()
        return self._pop_action()
