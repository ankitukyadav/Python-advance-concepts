"""
Advanced Python: Distributed Computing Framework
This script demonstrates building a distributed computing system using
multiprocessing, threading, and network communication.
"""

import multiprocessing as mp
import threading
import queue
import socket
import pickle
import time
import json
import hashlib
import uuid
from typing import Any, Dict, List, Callable, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import zmq
import asyncio
import aiohttp
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Message types and data structures
class MessageType(Enum):
    """Types of messages in the distributed system."""
    TASK_SUBMIT = "task_submit"
    TASK_RESULT = "task_result"
    TASK_ERROR = "task_error"
    WORKER_REGISTER = "worker_register"
    WORKER_HEARTBEAT = "worker_heartbeat"
    SYSTEM_STATUS = "system_status"
    SHUTDOWN = "shutdown"

@dataclass
class Task:
    """Represents a computational task."""
    id: str
    function_name: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    priority: int = 0
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

@dataclass
class TaskResult:
    """Represents the result of a task execution."""
    task_id: str
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    worker_id: str = ""

@dataclass
class WorkerInfo:
    """Information about a worker node."""
    id: str
    host: str
    port: int
    capabilities: List[str]
    max_concurrent_tasks: int
    current_tasks: int = 0
    last_heartbeat: float = 0.0
    status: str = "active"

@dataclass
class Message:
    """Generic message structure for distributed communication."""
    type: MessageType
    sender_id: str
    timestamp: float
    data: Dict[str, Any]
    message_id: str = ""
    
    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())

# Serialization utilities
class Serializer:
    """Handles serialization and deserialization of objects."""
    
    @staticmethod
    def serialize(obj: Any) -> bytes:
        """Serialize an object to bytes."""
        return pickle.dumps(obj)
    
    @staticmethod
    def deserialize(data: bytes) -> Any:
        """Deserialize bytes to an object."""
        return pickle.loads(data)
    
    @staticmethod
    def serialize_json(obj: Any) -> str:
        """Serialize an object to JSON string."""
        if hasattr(obj, '__dict__'):
            return json.dumps(asdict(obj) if hasattr(obj, '__dataclass_fields__') else obj.__dict__)
        return json.dumps(obj)
    
    @staticmethod
    def deserialize_json(data: str) -> Any:
        """Deserialize JSON string to an object."""
        return json.loads(data)

# Communication layer
class NetworkManager:
    """Manages network communication between nodes."""
    
    def __init__(self, host: str = "localhost", port: int = 5555):
        self.host = host
        self.port = port
        self.context = zmq.Context()
        self.socket = None
        self.running = False
    
    def start_server(self, handler: Callable[[Message], Optional[Message]]):
        """Start a ZeroMQ server."""
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{self.host}:{self.port}")
        self.running = True
        
        logger.info(f"Server started on {self.host}:{self.port}")
        
        while self.running:
            try:
                # Receive message
                data = self.socket.recv(zmq.NOBLOCK)
                message = Serializer.deserialize(data)
                
                # Process message
                response = handler(message)
                
                # Send response
                if response:
                    response_data = Serializer.serialize(response)
                    self.socket.send(response_data)
                else:
                    self.socket.send(b"ACK")
                    
            except zmq.Again:
                time.sleep(0.01)  # Non-blocking receive
            except Exception as e:
                logger.error(f"Server error: {e}")
                self.socket.send(Serializer.serialize({"error": str(e)}))
    
    def send_message(self, host: str, port: int, message: Message) -> Optional[Any]:
        """Send a message to a remote node."""
        client_socket = self.context.socket(zmq.REQ)
        client_socket.connect(f"tcp://{host}:{port}")
        
        try:
            # Send message
            data = Serializer.serialize(message)
            client_socket.send(data)
            
            # Receive response
            response_data = client_socket.recv()
            response = Serializer.deserialize(response_data)
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return None
        finally:
            client_socket.close()
    
    def stop(self):
        """Stop the network manager."""
        self.running = False
        if self.socket:
            self.socket.close()
        self.context.term()

# Task registry for function management
class TaskRegistry:
    """Registry for managing available tasks/functions."""
    
    def __init__(self):
        self.functions: Dict[str, Callable] = {}
        self.function_metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(self, name: str, func: Callable, metadata: Optional[Dict[str, Any]] = None):
        """Register a function for distributed execution."""
        self.functions[name] = func
        self.function_metadata[name] = metadata or {}
        logger.info(f"Registered function: {name}")
    
    def get_function(self, name: str) -> Optional[Callable]:
        """Get a registered function by name."""
        return self.functions.get(name)
    
    def list_functions(self) -> List[str]:
        """List all registered functions."""
        return list(self.functions.keys())
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a function."""
        return self.function_metadata.get(name, {})

# Worker node implementation
class WorkerNode:
    """A worker node that executes tasks."""
    
    def __init__(self, worker_id: str, host: str = "localhost", port: int = 5556,
                 max_concurrent_tasks: int = 4):
        self.worker_id = worker_id
        self.host = host
        self.port = port
        self.max_concurrent_tasks = max_concurrent_tasks
        self.current_tasks = 0
        self.task_registry = TaskRegistry()
        self.network_manager = NetworkManager(host, port)
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        
        # Register built-in functions
        self._register_builtin_functions()
    
    def _register_builtin_functions(self):
        """Register built-in functions."""
        
        def add_numbers(a: float, b: float) -> float:
            """Add two numbers."""
            time.sleep(0.1)  # Simulate work
            return a + b
        
        def multiply_matrix(matrix: List[List[float]], scalar: float) -> List[List[float]]:
            """Multiply matrix by scalar."""
            time.sleep(0.2)  # Simulate work
            return [[cell * scalar for cell in row] for row in matrix]
        
        def fibonacci(n: int) -> int:
            """Calculate Fibonacci number."""
            if n <= 1:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)
        
        def prime_check(n: int) -> bool:
            """Check if a number is prime."""
            if n < 2:
                return False
            for i in range(2, int(n ** 0.5) + 1):
                if n % i == 0:
                    return False
            return True
        
        self.task_registry.register("add_numbers", add_numbers)
        self.task_registry.register("multiply_matrix", multiply_matrix)
        self.task_registry.register("fibonacci", fibonacci)
        self.task_registry.register("prime_check", prime_check)
    
    def register_function(self, name: str, func: Callable, metadata: Optional[Dict[str, Any]] = None):
        """Register a custom function."""
        self.task_registry.register(name, func, metadata)
    
    def execute_task(self, task: Task) -> TaskResult:
        """Execute a single task."""
        start_time = time.time()
        
        try:
            func = self.task_registry.get_function(task.function_name)
            if not func:
                raise ValueError(f"Function '{task.function_name}' not found")
            
            # Execute function
            result = func(*task.args, **task.kwargs)
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task.id,
                result=result,
                execution_time=execution_time,
                worker_id=self.worker_id
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task {task.id} failed: {e}")
            
            return TaskResult(
                task_id=task.id,
                error=str(e),
                execution_time=execution_time,
                worker_id=self.worker_id
            )
    
    def handle_message(self, message: Message) -> Optional[Message]:
        """Handle incoming messages."""
        if message.type == MessageType.TASK_SUBMIT:
            task_data = message.data.get("task")
            if task_data:
                task = Task(**task_data)
                
                # Check if we can accept more tasks
                if self.current_tasks < self.max_concurrent_tasks:
                    # Submit task for execution
                    future = self.executor.submit(self.execute_task, task)
                    self.current_tasks += 1
                    
                    # Handle result asynchronously
                    def handle_result(fut):
                        try:
                            result = fut.result()
                            self.result_queue.put(result)
                        finally:
                            self.current_tasks -= 1
                    
                    future.add_done_callback(handle_result)
                    
                    return Message(
                        type=MessageType.TASK_RESULT,
                        sender_id=self.worker_id,
                        timestamp=time.time(),
                        data={"status": "accepted", "task_id": task.id}
                    )
                else:
                    return Message(
                        type=MessageType.TASK_ERROR,
                        sender_id=self.worker_id,
                        timestamp=time.time(),
                        data={"error": "Worker at capacity", "task_id": task.id}
                    )
        
        elif message.type == MessageType.SYSTEM_STATUS:
            return Message(
                type=MessageType.SYSTEM_STATUS,
                sender_id=self.worker_id,
                timestamp=time.time(),
                data={
                    "worker_info": {
                        "id": self.worker_id,
                        "host": self.host,
                        "port": self.port,
                        "current_tasks": self.current_tasks,
                        "max_concurrent_tasks": self.max_concurrent_tasks,
                        "available_functions": self.task_registry.list_functions()
                    }
                }
            )
        
        return None
    
    def start(self):
        """Start the worker node."""
        self.running = True
        logger.info(f"Starting worker {self.worker_id} on {self.host}:{self.port}")
        
        # Start network server in a separate thread
        server_thread = threading.Thread(
            target=self.network_manager.start_server,
            args=(self.handle_message,)
        )
        server_thread.daemon = True
        server_thread.start()
        
        # Main worker loop
        try:
            while self.running:
                time.sleep(1)  # Keep the worker alive
        except KeyboardInterrupt:
            logger.info("Worker shutting down...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the worker node."""
        self.running = False
        self.executor.shutdown(wait=True)
        self.network_manager.stop()

# Master node implementation
class MasterNode:
    """Master node that coordinates distributed tasks."""
    
    def __init__(self, host: str = "localhost", port: int = 5555):
        self.host = host
        self.port = port
        self.workers: Dict[str, WorkerInfo] = {}
        self.pending_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.task_queue = queue.PriorityQueue()
        self.network_manager = NetworkManager(host, port)
        self.running = False
        self.scheduler_thread = None
        
    def register_worker(self, worker_info: WorkerInfo):
        """Register a new worker."""
        self.workers[worker_info.id] = worker_info
        logger.info(f"Registered worker: {worker_info.id} at {worker_info.host}:{worker_info.port}")
    
    def submit_task(self, task: Task) -> str:
        """Submit a task for execution."""
        self.pending_tasks[task.id] = task
        # Use negative priority for max-heap behavior (higher priority first)
        self.task_queue.put((-task.priority, time.time(), task))
        logger.info(f"Submitted task: {task.id}")
        return task.id
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """Get the result of a completed task."""
        start_time = time.time()
        
        while True:
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            time.sleep(0.1)
    
    def find_available_worker(self, task: Task) -> Optional[WorkerInfo]:
        """Find an available worker for a task."""
        available_workers = [
            worker for worker in self.workers.values()
            if worker.status == "active" and worker.current_tasks < worker.max_concurrent_tasks
        ]
        
        if not available_workers:
            return None
        
        # Simple load balancing: choose worker with least current tasks
        return min(available_workers, key=lambda w: w.current_tasks)
    
    def schedule_tasks(self):
        """Task scheduler that runs in a separate thread."""
        while self.running:
            try:
                # Get next task from queue (with timeout to allow checking running flag)
                try:
                    _, _, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Find available worker
                worker = self.find_available_worker(task)
                if not worker:
                    # No available workers, put task back in queue
                    self.task_queue.put((-task.priority, time.time(), task))
                    time.sleep(0.5)
                    continue
                
                # Send task to worker
                message = Message(
                    type=MessageType.TASK_SUBMIT,
                    sender_id="master",
                    timestamp=time.time(),
                    data={"task": asdict(task)}
                )
                
                response = self.network_manager.send_message(worker.host, worker.port, message)
                
                if response and response.get("status") == "accepted":
                    worker.current_tasks += 1
                    logger.info(f"Task {task.id} assigned to worker {worker.id}")
                else:
                    # Task was rejected, put it back in queue
                    self.task_queue.put((-task.priority, time.time(), task))
                    logger.warning(f"Task {task.id} rejected by worker {worker.id}")
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(1)
    
    def handle_message(self, message: Message) -> Optional[Message]:
        """Handle incoming messages from workers."""
        if message.type == MessageType.WORKER_REGISTER:
            worker_data = message.data.get("worker_info")
            if worker_data:
                worker_info = WorkerInfo(**worker_data)
                self.register_worker(worker_info)
        
        elif message.type == MessageType.TASK_RESULT:
            task_id = message.data.get("task_id")
            result_data = message.data.get("result")
            
            if task_id and result_data:
                result = TaskResult(**result_data)
                self.completed_tasks[task_id] = result
                
                # Remove from pending tasks
                if task_id in self.pending_tasks:
                    del self.pending_tasks[task_id]
                
                # Update worker task count
                worker_id = result.worker_id
                if worker_id in self.workers:
                    self.workers[worker_id].current_tasks -= 1
                
                logger.info(f"Task {task_id} completed by worker {worker_id}")
        
        elif message.type == MessageType.WORKER_HEARTBEAT:
            worker_id = message.sender_id
            if worker_id in self.workers:
                self.workers[worker_id].last_heartbeat = time.time()
        
        return Message(
            type=MessageType.SYSTEM_STATUS,
            sender_id="master",
            timestamp=time.time(),
            data={"status": "ok"}
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "workers": len(self.workers),
            "active_workers": len([w for w in self.workers.values() if w.status == "active"]),
            "pending_tasks": len(self.pending_tasks),
            "completed_tasks": len(self.completed_tasks),
            "queue_size": self.task_queue.qsize(),
            "worker_details": [asdict(worker) for worker in self.workers.values()]
        }
    
    def start(self):
        """Start the master node."""
        self.running = True
        logger.info(f"Starting master node on {self.host}:{self.port}")
        
        # Start task scheduler
        self.scheduler_thread = threading.Thread(target=self.schedule_tasks)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        # Start network server
        try:
            self.network_manager.start_server(self.handle_message)
        except KeyboardInterrupt:
            logger.info("Master shutting down...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the master node."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        self.network_manager.stop()

# High-level distributed computing interface
class DistributedComputer:
    """High-level interface for distributed computing."""
    
    def __init__(self, master_host: str = "localhost", master_port: int = 5555):
        self.master_host = master_host
        self.master_port = master_port
        self.network_manager = NetworkManager()
    
    def submit_task(self, function_name: str, *args, priority: int = 0, 
                   timeout: Optional[float] = None, **kwargs) -> str:
        """Submit a task for distributed execution."""
        task = Task(
            id=str(uuid.uuid4()),
            function_name=function_name,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout
        )
        
        message = Message(
            type=MessageType.TASK_SUBMIT,
            sender_id="client",
            timestamp=time.time(),
            data={"task": asdict(task)}
        )
        
        response = self.network_manager.send_message(
            self.master_host, self.master_port, message
        )
        
        if response:
            return task.id
        else:
            raise RuntimeError("Failed to submit task")
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get the result of a task."""
        start_time = time.time()
        
        while True:
            message = Message(
                type=MessageType.SYSTEM_STATUS,
                sender_id="client",
                timestamp=time.time(),
                data={"query": "task_result", "task_id": task_id}
            )
            
            response = self.network_manager.send_message(
                self.master_host, self.master_port, message
            )
            
            if response and "result" in response:
                result_data = response["result"]
                if result_data.get("error"):
                    raise RuntimeError(result_data["error"])
                return result_data.get("result")
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} timed out")
            
            time.sleep(0.5)
    
    def map(self, function_name: str, iterable: List[Any], 
           chunk_size: Optional[int] = None) -> List[Any]:
        """Distributed map operation."""
        if chunk_size is None:
            chunk_size = max(1, len(iterable) // 10)  # Default to 10 chunks
        
        # Split iterable into chunks
        chunks = [iterable[i:i + chunk_size] for i in range(0, len(iterable), chunk_size)]
        
        # Submit tasks for each chunk
        task_ids = []
        for chunk in chunks:
            task_id = self.submit_task(f"map_{function_name}", chunk)
            task_ids.append(task_id)
        
        # Collect results
        results = []
        for task_id in task_ids:
            chunk_result = self.get_result(task_id)
            results.extend(chunk_result)
        
        return results
    
    def reduce(self, function_name: str, iterable: List[Any], 
              initial: Any = None) -> Any:
        """Distributed reduce operation."""
        # For simplicity, we'll do a tree reduction
        values = list(iterable)
        
        while len(values) > 1:
            # Pair up values and submit reduction tasks
            task_ids = []
            new_values = []
            
            for i in range(0, len(values), 2):
                if i + 1 < len(values):
                    # Pair reduction
                    task_id = self.submit_task(function_name, values[i], values[i + 1])
                    task_ids.append(task_id)
                else:
                    # Odd value, carry forward
                    new_values.append(values[i])
            
            # Collect results
            for task_id in task_ids:
                result = self.get_result(task_id)
                new_values.append(result)
            
            values = new_values
        
        return values[0] if values else initial

# Fault tolerance and monitoring
class HealthMonitor:
    """Monitors the health of the distributed system."""
    
    def __init__(self, master_host: str = "localhost", master_port: int = 5555):
        self.master_host = master_host
        self.master_port = master_port
        self.network_manager = NetworkManager()
        self.monitoring = False
        self.stats = {
            "uptime": 0,
            "total_tasks": 0,
            "failed_tasks": 0,
            "average_response_time": 0,
            "worker_failures": 0
        }
    
    def start_monitoring(self, interval: float = 10.0):
        """Start health monitoring."""
        self.monitoring = True
        start_time = time.time()
        
        def monitor_loop():
            while self.monitoring:
                try:
                    # Get system status
                    message = Message(
                        type=MessageType.SYSTEM_STATUS,
                        sender_id="monitor",
                        timestamp=time.time(),
                        data={"query": "full_status"}
                    )
                    
                    response_start = time.time()
                    response = self.network_manager.send_message(
                        self.master_host, self.master_port, message
                    )
                    response_time = time.time() - response_start
                    
                    if response:
                        self.stats["uptime"] = time.time() - start_time
                        self.stats["average_response_time"] = response_time
                        
                        # Log system status
                        status = response.get("system_status", {})
                        logger.info(f"System Status: {status}")
                        
                        # Check for worker failures
                        workers = status.get("worker_details", [])
                        for worker in workers:
                            if worker.get("status") != "active":
                                self.stats["worker_failures"] += 1
                                logger.warning(f"Worker {worker.get('id')} is not active")
                    
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                
                time.sleep(interval)
        
        monitor_thread = threading.Thread(target=monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return self.stats.copy()

# Load balancing strategies
class LoadBalancer:
    """Load balancing strategies for task distribution."""
    
    @staticmethod
    def round_robin(workers: List[WorkerInfo], task: Task) -> Optional[WorkerInfo]:
        """Round-robin load balancing."""
        available_workers = [w for w in workers if w.current_tasks < w.max_concurrent_tasks]
        if not available_workers:
            return None
        
        # Simple round-robin based on worker ID
        return min(available_workers, key=lambda w: w.id)
    
    @staticmethod
    def least_loaded(workers: List[WorkerInfo], task: Task) -> Optional[WorkerInfo]:
        """Least loaded worker selection."""
        available_workers = [w for w in workers if w.current_tasks < w.max_concurrent_tasks]
        if not available_workers:
            return None
        
        return min(available_workers, key=lambda w: w.current_tasks / w.max_concurrent_tasks)
    
    @staticmethod
    def capability_based(workers: List[WorkerInfo], task: Task) -> Optional[WorkerInfo]:
        """Capability-based worker selection."""
        # Filter workers that can handle the task
        capable_workers = []
        for worker in workers:
            if (worker.current_tasks < worker.max_concurrent_tasks and
                task.function_name in worker.capabilities):
                capable_workers.append(worker)
        
        if not capable_workers:
            return None
        
        # Among capable workers, choose least loaded
        return min(capable_workers, key=lambda w: w.current_tasks / w.max_concurrent_tasks)

# Caching layer
class DistributedCache:
    """Distributed caching system."""
    
    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.cache_stats = {"hits": 0, "misses": 0}
        self.lock = threading.RLock()
    
    def _generate_key(self, function_name: str, args: Tuple, kwargs: Dict) -> str:
        """Generate cache key from function name and arguments."""
        key_data = {
            "function": function_name,
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, function_name: str, args: Tuple, kwargs: Dict) -> Optional[Any]:
        """Get cached result."""
        key = self._generate_key(function_name, args, kwargs)
        
        with self.lock:
            if key in self.cache:
                self.cache_stats["hits"] += 1
                return self.cache[key]
            else:
                self.cache_stats["misses"] += 1
                return None
    
    def put(self, function_name: str, args: Tuple, kwargs: Dict, result: Any):
        """Cache a result."""
        key = self._generate_key(function_name, args, kwargs)
        
        with self.lock:
            self.cache[key] = result
    
    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.cache_stats = {"hits": 0, "misses": 0}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
            hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "hits": self.cache_stats["hits"],
                "misses": self.cache_stats["misses"],
                "hit_rate": hit_rate
            }

# Example applications
def create_computation_tasks():
    """Create example computation tasks."""
    tasks = []
    
    # Mathematical computations
    for i in range(10, 20):
        task = Task(
            id=f"fib_{i}",
            function_name="fibonacci",
            args=(i,),
            kwargs={},
            priority=1
        )
        tasks.append(task)
    
    # Prime number checks
    for i in range(1000, 1100, 10):
        task = Task(
            id=f"prime_{i}",
            function_name="prime_check",
            args=(i,),
            kwargs={},
            priority=2
        )
        tasks.append(task)
    
    # Matrix operations
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    for scalar in [2, 3, 4, 5]:
        task = Task(
            id=f"matrix_{scalar}",
            function_name="multiply_matrix",
            args=(matrix, scalar),
            kwargs={},
            priority=0
        )
        tasks.append(task)
    
    return tasks

def demonstrate_basic_distributed_computing():
    """Demonstrate basic distributed computing."""
    print("=== Basic Distributed Computing Demo ===")
    
    # Start master node in a separate process
    def run_master():
        master = MasterNode()
        master.start()
    
    # Start worker nodes in separate processes
    def run_worker(worker_id: str, port: int):
        worker = WorkerNode(worker_id, port=port)
        worker.start()
    
    # For demonstration, we'll simulate this without actually starting processes
    print("In a real scenario, you would start:")
    print("1. Master node on port 5555")
    print("2. Multiple worker nodes on different ports")
    print("3. Submit tasks through DistributedComputer interface")
    
    # Simulate task submission and results
    tasks = create_computation_tasks()
    print(f"\nCreated {len(tasks)} example tasks:")
    for task in tasks[:5]:  # Show first 5
        print(f"  - {task.id}: {task.function_name}({task.args})")
    
    print("\nTask execution would be distributed across available workers...")

def demonstrate_load_balancing():
    """Demonstrate load balancing strategies."""
    print("\n=== Load Balancing Demo ===")
    
    # Create mock workers
    workers = [
        WorkerInfo("worker1", "localhost", 5556, ["fibonacci", "prime_check"], 4, 2),
        WorkerInfo("worker2", "localhost", 5557, ["multiply_matrix", "add_numbers"], 2, 0),
        WorkerInfo("worker3", "localhost", 5558, ["fibonacci", "multiply_matrix"], 6, 4),
    ]
    
    # Create test task
    task = Task("test", "fibonacci", (15,), {})
    
    # Test different load balancing strategies
    strategies = {
        "Round Robin": LoadBalancer.round_robin,
        "Least Loaded": LoadBalancer.least_loaded,
        "Capability Based": LoadBalancer.capability_based
    }
    
    for name, strategy in strategies.items():
        selected_worker = strategy(workers, task)
        if selected_worker:
            load_ratio = selected_worker.current_tasks / selected_worker.max_concurrent_tasks
            print(f"{name}: Selected {selected_worker.id} (load: {load_ratio:.2f})")
        else:
            print(f"{name}: No available worker")

def demonstrate_caching():
    """Demonstrate distributed caching."""
    print("\n=== Distributed Caching Demo ===")
    
    cache = DistributedCache()
    
    # Simulate function calls
    test_calls = [
        ("fibonacci", (10,), {}),
        ("fibonacci", (10,), {}),  # Cache hit
        ("prime_check", (97,), {}),
        ("fibonacci", (15,), {}),
        ("prime_check", (97,), {}),  # Cache hit
    ]
    
    for func_name, args, kwargs in test_calls:
        # Check cache first
        cached_result = cache.get(func_name, args, kwargs)
        
        if cached_result is not None:
            print(f"Cache HIT: {func_name}{args} = {cached_result}")
        else:
            print(f"Cache MISS: {func_name}{args}")
            # Simulate computation
            if func_name == "fibonacci":
                result = args[0] * 2  # Simplified result
            else:
                result = True  # Simplified result
            
            cache.put(func_name, args, kwargs, result)
            print(f"  Computed and cached: {result}")
    
    # Show cache statistics
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Size: {stats['size']} entries")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Hits: {stats['hits']}, Misses: {stats['misses']}")

def demonstrate_fault_tolerance():
    """Demonstrate fault tolerance mechanisms."""
    print("\n=== Fault Tolerance Demo ===")
    
    # Simulate worker failures and recovery
    workers = {
        "worker1": WorkerInfo("worker1", "localhost", 5556, ["fibonacci"], 4, 2, time.time(), "active"),
        "worker2": WorkerInfo("worker2", "localhost", 5557, ["prime_check"], 2, 1, time.time() - 30, "inactive"),
        "worker3": WorkerInfo("worker3", "localhost", 5558, ["multiply_matrix"], 6, 0, time.time(), "active"),
    }
    
    print("Worker Status:")
    for worker_id, worker in workers.items():
        status = "HEALTHY" if worker.status == "active" and (time.time() - worker.last_heartbeat) < 15 else "FAILED"
        print(f"  {worker_id}: {status} (last heartbeat: {time.time() - worker.last_heartbeat:.1f}s ago)")
    
    # Simulate task retry mechanism
    failed_task = Task("retry_test", "fibonacci", (20,), {}, retry_count=2, max_retries=3)
    
    print(f"\nTask Retry Simulation:")
    print(f"  Task: {failed_task.id}")
    print(f"  Current retry count: {failed_task.retry_count}/{failed_task.max_retries}")
    
    if failed_task.retry_count < failed_task.max_retries:
        print("  → Task will be retried on another worker")
    else:
        print("  → Task has exceeded max retries, marking as failed")

def main():
    """Main demonstration function."""
    print("=== Advanced Python: Distributed Computing Framework Demo ===")
    
    demonstrate_basic_distributed_computing()
    demonstrate_load_balancing()
    demonstrate_caching()
    demonstrate_fault_tolerance()
    
    print("\n=== Demo Complete ===")
    print("\nTo run a real distributed system:")
    print("1. Start master: python -c 'from distributed_computing import MasterNode; MasterNode().start()'")
    print("2. Start workers: python -c 'from distributed_computing import WorkerNode; WorkerNode(\"worker1\", port=5556).start()'")
    print("3. Submit tasks: python -c 'from distributed_computing import DistributedComputer; dc = DistributedComputer(); print(dc.submit_task(\"fibonacci\", 20))'")

if __name__ == "__main__":
    main()
