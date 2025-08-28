"""
Advanced Python: Performance Optimization Techniques
This script demonstrates various performance optimization techniques including
profiling, caching, vectorization, and memory optimization.
"""

import time
import cProfile
import pstats
import io
import sys
import gc
import tracemalloc
import functools
import weakref
import array
import numpy as np
from typing import Any, Dict, List, Callable, Optional, Tuple, Iterator
from collections import defaultdict, deque
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import threading
from contextlib import contextmanager
import psutil
import memory_profiler

# Profiling utilities
class PerformanceProfiler:
    """Advanced performance profiling utilities."""
    
    def __init__(self):
        self.profiles = {}
        self.memory_snapshots = []
    
    @contextmanager
    def profile_time(self, name: str):
        """Context manager for timing code blocks."""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            self.profiles[name] = {
                'execution_time': execution_time,
                'memory_delta': memory_delta,
                'start_memory': start_memory,
                'end_memory': end_memory
            }
            
            print(f"{name}: {execution_time:.4f}s, Memory: {memory_delta:+.2f}MB")
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator for profiling functions."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.profile_time(func.__name__):
                return func(*args, **kwargs)
        return wrapper
    
    def profile_memory_detailed(self, func: Callable) -> Callable:
        """Detailed memory profiling decorator."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracemalloc.start()
            
            result = func(*args, **kwargs)
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            print(f"{func.__name__} - Current memory: {current / 1024 / 1024:.2f}MB, "
                  f"Peak memory: {peak / 1024 / 1024:.2f}MB")
            
            return result
        return wrapper
    
    def profile_cpu_detailed(self, func: Callable, sort_by: str = 'cumulative') -> str:
        """Detailed CPU profiling."""
        profiler = cProfile.Profile()
        
        profiler.enable()
        result = func()
        profiler.disable()
        
        # Capture profiling output
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats(sort_by)
        ps.print_stats()
        
        return s.getvalue()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary."""
        if not self.profiles:
            return {}
        
        total_time = sum(p['execution_time'] for p in self.profiles.values())
        total_memory = sum(p['memory_delta'] for p in self.profiles.values())
        
        return {
            'total_execution_time': total_time,
            'total_memory_delta': total_memory,
            'function_profiles': self.profiles,
            'slowest_function': max(self.profiles.items(), key=lambda x: x[1]['execution_time']),
            'memory_heaviest': max(self.profiles.items(), key=lambda x: x[1]['memory_delta'])
        }

# Caching strategies
class LRUCache:
    """Custom LRU (Least Recently Used) cache implementation."""
    
    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self.cache = {}
        self.access_order = deque()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: Any) -> Any:
        """Get value from cache."""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, key: Any, value: Any) -> None:
        """Put value in cache."""
        if key in self.cache:
            # Update existing key
            self.access_order.remove(key)
        elif len(self.cache) >= self.maxsize:
            # Remove least recently used
            lru_key = self.access_order.popleft()
            del self.cache[lru_key]
        
        self.cache[key] = value
        self.access_order.append(key)
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.access_order.clear()
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'maxsize': self.maxsize,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }

class MemoizationCache:
    """Advanced memoization with different strategies."""
    
    def __init__(self, strategy: str = 'lru', maxsize: int = 128):
        self.strategy = strategy
        self.maxsize = maxsize
        
        if strategy == 'lru':
            self.cache = LRUCache(maxsize)
        else:
            self.cache = {}
    
    def memoize(self, func: Callable) -> Callable:
        """Memoization decorator."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = self._make_key(args, kwargs)
            
            # Check cache
            if self.strategy == 'lru':
                result = self.cache.get(key)
                if result is not None:
                    return result
            else:
                if key in self.cache:
                    return self.cache[key]
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            if self.strategy == 'lru':
                self.cache.put(key, result)
            else:
                if len(self.cache) < self.maxsize:
                    self.cache[key] = result
            
            return result
        
        wrapper.cache = self.cache
        wrapper.cache_clear = self.cache.clear if hasattr(self.cache, 'clear') else lambda: self.cache.clear()
        return wrapper
    
    def _make_key(self, args: Tuple, kwargs: Dict) -> str:
        """Create cache key from arguments."""
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return "|".join(key_parts)

# Memory optimization techniques
class MemoryOptimizer:
    """Memory optimization utilities."""
    
    @staticmethod
    def use_slots_example():
        """Demonstrate __slots__ for memory optimization."""
        
        # Regular class
        class RegularPoint:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        
        # Optimized class with __slots__
        class OptimizedPoint:
            __slots__ = ['x', 'y']
            
            def __init__(self, x, y):
                self.x = x
                self.y = y
        
        return RegularPoint, OptimizedPoint
    
    @staticmethod
    def memory_efficient_data_structures():
        """Demonstrate memory-efficient data structures."""
        
        # Use array instead of list for numeric data
        regular_list = [i for i in range(1000000)]
        efficient_array = array.array('i', range(1000000))
        
        # Use generators instead of lists when possible
        def number_generator(n):
            for i in range(n):
                yield i * i
        
        # Use __slots__ and weak references
        class MemoryEfficientNode:
            __slots__ = ['value', '_parent_ref', 'children']
            
            def __init__(self, value):
                self.value = value
                self._parent_ref = None
                self.children = []
            
            @property
            def parent(self):
                return self._parent_ref() if self._parent_ref else None
            
            @parent.setter
            def parent(self, parent):
                self._parent_ref = weakref.ref(parent) if parent else None
        
        return {
            'regular_list_size': sys.getsizeof(regular_list),
            'efficient_array_size': sys.getsizeof(efficient_array),
            'generator': number_generator,
            'efficient_node': MemoryEfficientNode
        }
    
    @staticmethod
    def optimize_string_operations():
        """Demonstrate string optimization techniques."""
        
        # Use join instead of concatenation
        def inefficient_string_concat(strings):
            result = ""
            for s in strings:
                result += s
            return result
        
        def efficient_string_join(strings):
            return "".join(strings)
        
        # Use string formatting efficiently
        def efficient_formatting(name, age, city):
            return f"Name: {name}, Age: {age}, City: {city}"
        
        # Use string interning for repeated strings
        def string_interning_example():
            # These will be the same object in memory
            s1 = sys.intern("repeated_string")
            s2 = sys.intern("repeated_string")
            return s1 is s2
        
        return {
            'inefficient_concat': inefficient_string_concat,
            'efficient_join': efficient_string_join,
            'efficient_formatting': efficient_formatting,
            'interning_works': string_interning_example()
        }

# Vectorization and NumPy optimization
class VectorizationOptimizer:
    """Vectorization optimization techniques."""
    
    @staticmethod
    def compare_loop_vs_vectorized():
        """Compare loop-based vs vectorized operations."""
        
        # Generate test data
        size = 1000000
        a = np.random.rand(size)
        b = np.random.rand(size)
        
        # Loop-based approach
        def loop_based_operation(arr1, arr2):
            result = np.zeros(len(arr1))
            for i in range(len(arr1)):
                result[i] = arr1[i] * arr2[i] + np.sin(arr1[i])
            return result
        
        # Vectorized approach
        def vectorized_operation(arr1, arr2):
            return arr1 * arr2 + np.sin(arr1)
        
        return {
            'data_size': size,
            'loop_based': loop_based_operation,
            'vectorized': vectorized_operation,
            'test_data': (a, b)
        }
    
    @staticmethod
    def numpy_optimization_tips():
        """Demonstrate NumPy optimization techniques."""
        
        # Pre-allocate arrays
        def preallocate_arrays(size):
            # Efficient: pre-allocate
            result = np.empty(size)
            for i in range(size):
                result[i] = i ** 2
            return result
        
        # Use appropriate data types
        def use_appropriate_dtypes():
            # Use smaller data types when possible
            large_int_array = np.array([1, 2, 3, 4, 5], dtype=np.int64)
            small_int_array = np.array([1, 2, 3, 4, 5], dtype=np.int8)
            
            return {
                'large_size': large_int_array.nbytes,
                'small_size': small_int_array.nbytes
            }
        
        # Use views instead of copies
        def use_views_not_copies():
            original = np.arange(1000000)
            
            # This creates a view (efficient)
            view = original[::2]
            
            # This creates a copy (less efficient)
            copy = original[::2].copy()
            
            return {
                'original_size': original.nbytes,
                'view_shares_memory': np.shares_memory(original, view),
                'copy_shares_memory': np.shares_memory(original, copy)
            }
        
        return {
            'preallocate': preallocate_arrays,
            'dtypes': use_appropriate_dtypes(),
            'views': use_views_not_copies()
        }

# Parallel processing optimization
class ParallelOptimizer:
    """Parallel processing optimization techniques."""
    
    def __init__(self):
        self.cpu_count = mp.cpu_count()
    
    def compare_sequential_vs_parallel(self, task_func: Callable, data: List[Any]) -> Dict[str, Any]:
        """Compare sequential vs parallel execution."""
        
        # Sequential execution
        start_time = time.perf_counter()
        sequential_results = [task_func(item) for item in data]
        sequential_time = time.perf_counter() - start_time
        
        # Parallel execution with threads (I/O bound tasks)
        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.cpu_count) as executor:
            thread_results = list(executor.map(task_func, data))
        thread_time = time.perf_counter() - start_time
        
        # Parallel execution with processes (CPU bound tasks)
        start_time = time.perf_counter()
        with ProcessPoolExecutor(max_workers=self.cpu_count) as executor:
            process_results = list(executor.map(task_func, data))
        process_time = time.perf_counter() - start_time
        
        return {
            'sequential_time': sequential_time,
            'thread_time': thread_time,
            'process_time': process_time,
            'thread_speedup': sequential_time / thread_time,
            'process_speedup': sequential_time / process_time,
            'results_match': (sequential_results == thread_results == process_results)
        }
    
    def optimize_data_sharing(self):
        """Demonstrate efficient data sharing between processes."""
        
        # Shared memory array
        def use_shared_memory():
            # Create shared array
            shared_array = mp.Array('d', range(1000000))
            
            def worker_func(shared_arr, start, end):
                # Access shared memory
                for i in range(start, end):
                    shared_arr[i] = shared_arr[i] ** 2
            
            # Split work among processes
            chunk_size = len(shared_array) // self.cpu_count
            processes = []
            
            for i in range(self.cpu_count):
                start = i * chunk_size
                end = start + chunk_size if i < self.cpu_count - 1 else len(shared_array)
                p = mp.Process(target=worker_func, args=(shared_array, start, end))
                processes.append(p)
                p.start()
            
            for p in processes:
                p.join()
            
            return list(shared_array[:10])  # Return first 10 elements
        
        return use_shared_memory

# Algorithm optimization
class AlgorithmOptimizer:
    """Algorithm-level optimization techniques."""
    
    @staticmethod
    def optimize_search_algorithms():
        """Compare different search algorithm implementations."""
        
        # Linear search
        def linear_search(arr, target):
            for i, value in enumerate(arr):
                if value == target:
                    return i
            return -1
        
        # Binary search (requires sorted array)
        def binary_search(arr, target):
            left, right = 0, len(arr) - 1
            
            while left <= right:
                mid = (left + right) // 2
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            return -1
        
        # Hash-based search
        def hash_search(arr, target):
                        hash_map = {value: i for i, value in enumerate(arr)}
            return hash_map.get(target, -1)
        
        return {
            'linear_search': linear_search,
            'binary_search': binary_search,
            'hash_search': hash_search
        }
    
    @staticmethod
    def optimize_sorting_algorithms():
        """Compare different sorting algorithm implementations."""
        
        # Quick sort implementation
        def quicksort(arr):
            if len(arr) <= 1:
                return arr
            
            pivot = arr[len(arr) // 2]
            left = [x for x in arr if x < pivot]
            middle = [x for x in arr if x == pivot]
            right = [x for x in arr if x > pivot]
            
            return quicksort(left) + middle + quicksort(right)
        
        # Merge sort implementation
        def mergesort(arr):
            if len(arr) <= 1:
                return arr
            
            mid = len(arr) // 2
            left = mergesort(arr[:mid])
            right = mergesort(arr[mid:])
            
            return merge(left, right)
        
        def merge(left, right):
            result = []
            i = j = 0
            
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1
            
            result.extend(left[i:])
            result.extend(right[j:])
            return result
        
        # Heap sort implementation
        def heapsort(arr):
            def heapify(arr, n, i):
                largest = i
                left = 2 * i + 1
                right = 2 * i + 2
                
                if left < n and arr[left] > arr[largest]:
                    largest = left
                
                if right < n and arr[right] > arr[largest]:
                    largest = right
                
                if largest != i:
                    arr[i], arr[largest] = arr[largest], arr[i]
                    heapify(arr, n, largest)
            
            n = len(arr)
            
            # Build max heap
            for i in range(n // 2 - 1, -1, -1):
                heapify(arr, n, i)
            
            # Extract elements from heap
            for i in range(n - 1, 0, -1):
                arr[0], arr[i] = arr[i], arr[0]
                heapify(arr, i, 0)
            
            return arr
        
        return {
            'quicksort': quicksort,
            'mergesort': mergesort,
            'heapsort': heapsort
        }
    
    @staticmethod
    def optimize_data_structures():
        """Demonstrate optimized data structure implementations."""
        
        # Optimized deque for frequent insertions/deletions
        class OptimizedDeque:
            def __init__(self):
                self.items = deque()
            
            def append_left(self, item):
                self.items.appendleft(item)
            
            def append_right(self, item):
                self.items.append(item)
            
            def pop_left(self):
                return self.items.popleft() if self.items else None
            
            def pop_right(self):
                return self.items.pop() if self.items else None
        
        # Optimized dictionary with default values
        class OptimizedCounter:
            def __init__(self):
                self.counts = defaultdict(int)
            
            def increment(self, key):
                self.counts[key] += 1
            
            def get_count(self, key):
                return self.counts[key]
            
            def most_common(self, n=None):
                sorted_items = sorted(self.counts.items(), key=lambda x: x[1], reverse=True)
                return sorted_items[:n] if n else sorted_items
        
        # Memory-efficient set operations
        class OptimizedSet:
            def __init__(self, items=None):
                self.items = set(items) if items else set()
            
            def union_efficient(self, other):
                # Use |= for in-place union (more memory efficient)
                result = OptimizedSet(self.items)
                result.items |= other.items
                return result
            
            def intersection_efficient(self, other):
                # Use &= for in-place intersection
                result = OptimizedSet(self.items)
                result.items &= other.items
                return result
        
        return {
            'optimized_deque': OptimizedDeque,
            'optimized_counter': OptimizedCounter,
            'optimized_set': OptimizedSet
        }

# I/O optimization
class IOOptimizer:
    """I/O operation optimization techniques."""
    
    @staticmethod
    def optimize_file_operations():
        """Demonstrate optimized file I/O operations."""
        
        # Buffered reading
        def read_file_buffered(filename, buffer_size=8192):
            content = []
            with open(filename, 'r', buffering=buffer_size) as f:
                while True:
                    chunk = f.read(buffer_size)
                    if not chunk:
                        break
                    content.append(chunk)
            return ''.join(content)
        
        # Memory-mapped file reading
        def read_file_mmap(filename):
            import mmap
            with open(filename, 'r') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                    return mmapped_file.read().decode('utf-8')
        
        # Batch writing
        def write_file_batched(filename, data_generator, batch_size=1000):
            with open(filename, 'w') as f:
                batch = []
                for item in data_generator:
                    batch.append(str(item))
                    if len(batch) >= batch_size:
                        f.write('\n'.join(batch) + '\n')
                        batch = []
                
                # Write remaining items
                if batch:
                    f.write('\n'.join(batch) + '\n')
        
        return {
            'buffered_read': read_file_buffered,
            'mmap_read': read_file_mmap,
            'batched_write': write_file_batched
        }
    
    @staticmethod
    def optimize_network_operations():
        """Demonstrate optimized network I/O."""
        
        # Connection pooling simulation
        class ConnectionPool:
            def __init__(self, max_connections=10):
                self.max_connections = max_connections
                self.available_connections = deque()
                self.active_connections = set()
                self.lock = threading.Lock()
            
            def get_connection(self):
                with self.lock:
                    if self.available_connections:
                        conn = self.available_connections.popleft()
                        self.active_connections.add(conn)
                        return conn
                    elif len(self.active_connections) < self.max_connections:
                        conn = f"connection_{len(self.active_connections)}"
                        self.active_connections.add(conn)
                        return conn
                    else:
                        return None  # Pool exhausted
            
            def return_connection(self, conn):
                with self.lock:
                    if conn in self.active_connections:
                        self.active_connections.remove(conn)
                        self.available_connections.append(conn)
        
        # Batch request processing
        def batch_requests(requests, batch_size=10):
            batches = []
            for i in range(0, len(requests), batch_size):
                batch = requests[i:i + batch_size]
                batches.append(batch)
            return batches
        
        return {
            'connection_pool': ConnectionPool,
            'batch_requests': batch_requests
        }

# Comprehensive optimization benchmark
class OptimizationBenchmark:
    """Comprehensive benchmark for optimization techniques."""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.results = {}
    
    def benchmark_caching(self):
        """Benchmark different caching strategies."""
        print("=== Caching Benchmark ===")
        
        # Test function (expensive computation)
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)
        
        # Create cached versions
        lru_cache = MemoizationCache('lru', 100)
        cached_fib = lru_cache.memoize(fibonacci)
        
        # Benchmark
        test_values = [20, 25, 20, 30, 25, 20]  # Repeated values to test cache hits
        
        # Without caching
        with self.profiler.profile_time("fibonacci_no_cache"):
            no_cache_results = [fibonacci(n) for n in test_values]
        
        # With caching
        with self.profiler.profile_time("fibonacci_with_cache"):
            cache_results = [cached_fib(n) for n in test_values]
        
        cache_stats = cached_fib.cache.stats() if hasattr(cached_fib.cache, 'stats') else {}
        
        self.results['caching'] = {
            'no_cache_time': self.profiler.profiles['fibonacci_no_cache']['execution_time'],
            'cache_time': self.profiler.profiles['fibonacci_with_cache']['execution_time'],
            'cache_stats': cache_stats,
            'results_match': no_cache_results == cache_results
        }
    
    def benchmark_vectorization(self):
        """Benchmark vectorization vs loops."""
        print("=== Vectorization Benchmark ===")
        
        optimizer = VectorizationOptimizer()
        comparison = optimizer.compare_loop_vs_vectorized()
        
        a, b = comparison['test_data']
        
        # Benchmark loop-based
        with self.profiler.profile_time("loop_based_operation"):
            loop_result = comparison['loop_based'](a, b)
        
        # Benchmark vectorized
        with self.profiler.profile_time("vectorized_operation"):
            vectorized_result = comparison['vectorized'](a, b)
        
        self.results['vectorization'] = {
            'loop_time': self.profiler.profiles['loop_based_operation']['execution_time'],
            'vectorized_time': self.profiler.profiles['vectorized_operation']['execution_time'],
            'speedup': self.profiler.profiles['loop_based_operation']['execution_time'] / 
                      self.profiler.profiles['vectorized_operation']['execution_time'],
            'results_close': np.allclose(loop_result, vectorized_result)
        }
    
    def benchmark_parallel_processing(self):
        """Benchmark parallel vs sequential processing."""
        print("=== Parallel Processing Benchmark ===")
        
        # CPU-intensive task
        def cpu_intensive_task(n):
            total = 0
            for i in range(n):
                total += i ** 2
            return total
        
        # Test data
        test_data = [100000] * 8  # 8 tasks
        
        optimizer = ParallelOptimizer()
        results = optimizer.compare_sequential_vs_parallel(cpu_intensive_task, test_data)
        
        self.results['parallel_processing'] = results
    
    def benchmark_memory_optimization(self):
        """Benchmark memory optimization techniques."""
        print("=== Memory Optimization Benchmark ===")
        
        # Test __slots__ vs regular class
        RegularPoint, OptimizedPoint = MemoryOptimizer.use_slots_example()
        
        # Create many instances
        n_instances = 100000
        
        with self.profiler.profile_time("regular_class_creation"):
            regular_points = [RegularPoint(i, i+1) for i in range(n_instances)]
        
        with self.profiler.profile_time("optimized_class_creation"):
            optimized_points = [OptimizedPoint(i, i+1) for i in range(n_instances)]
        
        # Memory usage comparison
        regular_memory = sys.getsizeof(regular_points[0].__dict__) if hasattr(regular_points[0], '__dict__') else 0
        optimized_memory = sys.getsizeof(optimized_points[0])
        
        self.results['memory_optimization'] = {
            'regular_creation_time': self.profiler.profiles['regular_class_creation']['execution_time'],
            'optimized_creation_time': self.profiler.profiles['optimized_class_creation']['execution_time'],
            'regular_instance_memory': regular_memory,
            'optimized_instance_memory': optimized_memory,
            'memory_savings': regular_memory - optimized_memory if regular_memory > 0 else 0
        }
    
    def benchmark_algorithms(self):
        """Benchmark different algorithm implementations."""
        print("=== Algorithm Benchmark ===")
        
        # Test sorting algorithms
        optimizer = AlgorithmOptimizer()
        sorting_funcs = optimizer.optimize_sorting_algorithms()
        
        # Generate test data
        test_data = list(np.random.randint(0, 10000, 10000))
        
        # Benchmark each sorting algorithm
        for name, sort_func in sorting_funcs.items():
            test_copy = test_data.copy()
            with self.profiler.profile_time(f"sort_{name}"):
                sorted_result = sort_func(test_copy)
        
        # Test search algorithms
        search_funcs = optimizer.optimize_search_algorithms()
        sorted_data = sorted(test_data)
        target = sorted_data[len(sorted_data) // 2]  # Middle element
        
        for name, search_func in search_funcs.items():
            if name == 'binary_search':
                search_data = sorted_data
            else:
                search_data = test_data
            
            with self.profiler.profile_time(f"search_{name}"):
                for _ in range(1000):  # Multiple searches for better timing
                    result = search_func(search_data, target)
        
        # Extract algorithm results
        algorithm_results = {}
        for profile_name, profile_data in self.profiler.profiles.items():
            if profile_name.startswith(('sort_', 'search_')):
                algorithm_results[profile_name] = profile_data['execution_time']
        
        self.results['algorithms'] = algorithm_results
    
    def run_full_benchmark(self):
        """Run complete optimization benchmark suite."""
        print("=== Running Full Optimization Benchmark ===\n")
        
        self.benchmark_caching()
        self.benchmark_vectorization()
        self.benchmark_parallel_processing()
        self.benchmark_memory_optimization()
        self.benchmark_algorithms()
        
        return self.get_benchmark_summary()
    
    def get_benchmark_summary(self):
        """Get comprehensive benchmark summary."""
        summary = {
            'profiler_summary': self.profiler.get_summary(),
            'optimization_results': self.results
        }
        
        # Calculate overall performance improvements
        improvements = {}
        
        if 'caching' in self.results:
            cache_data = self.results['caching']
            if cache_data['no_cache_time'] > 0:
                improvements['caching_speedup'] = cache_data['no_cache_time'] / cache_data['cache_time']
        
        if 'vectorization' in self.results:
            improvements['vectorization_speedup'] = self.results['vectorization']['speedup']
        
        if 'parallel_processing' in self.results:
            parallel_data = self.results['parallel_processing']
            improvements['thread_speedup'] = parallel_data['thread_speedup']
            improvements['process_speedup'] = parallel_data['process_speedup']
        
        summary['performance_improvements'] = improvements
        
        return summary

# Demonstration functions
def demonstrate_profiling():
    """Demonstrate profiling techniques."""
    print("=== Profiling Demonstration ===")
    
    profiler = PerformanceProfiler()
    
    @profiler.profile_function
    def slow_function():
        time.sleep(0.1)
        return sum(i ** 2 for i in range(10000))
    
    @profiler.profile_memory_detailed
    def memory_intensive_function():
        # Create large data structures
        data = [i ** 2 for i in range(100000)]
        return sum(data)
    
    # Run profiled functions
    result1 = slow_function()
    result2 = memory_intensive_function()
    
    # Show profiling summary
    summary = profiler.get_summary()
    print(f"Profiling Summary:")
    print(f"  Total execution time: {summary['total_execution_time']:.4f}s")
    print(f"  Total memory delta: {summary['total_memory_delta']:.2f}MB")
    print(f"  Slowest function: {summary['slowest_function'][0]}")

def demonstrate_caching_strategies():
    """Demonstrate different caching strategies."""
    print("\n=== Caching Strategies Demonstration ===")
    
    # LRU Cache demonstration
    lru_cache = LRUCache(maxsize=3)
    
    # Add items to cache
    for i in range(5):
        lru_cache.put(f"key_{i}", f"value_{i}")
        print(f"Added key_{i}, cache size: {len(lru_cache.cache)}")
    
    # Access items (this affects LRU order)
    print(f"Accessing key_2: {lru_cache.get('key_2')}")
    print(f"Accessing key_4: {lru_cache.get('key_4')}")
    
    # Show cache stats
    stats = lru_cache.stats()
    print(f"Cache stats: {stats}")
    
    # Memoization demonstration
    memo_cache = MemoizationCache('lru', 50)
    
    @memo_cache.memoize
    def expensive_function(n):
        time.sleep(0.01)  # Simulate expensive computation
        return n ** 2
    
    # Test memoization
    print("\nMemoization test:")
    for i in [5, 3, 5, 7, 3]:  # Repeated values
        start = time.time()
        result = expensive_function(i)
        elapsed = time.time() - start
        print(f"f({i}) = {result}, time: {elapsed:.4f}s")

def demonstrate_memory_optimization():
    """Demonstrate memory optimization techniques."""
    print("\n=== Memory Optimization Demonstration ===")
    
    optimizer = MemoryOptimizer()
    
    # Demonstrate __slots__
    RegularPoint, OptimizedPoint = optimizer.use_slots_example()
    
    regular_point = RegularPoint(1, 2)
    optimized_point = OptimizedPoint(1, 2)
    
    print(f"Regular point memory: {sys.getsizeof(regular_point)} bytes")
    print(f"Optimized point memory: {sys.getsizeof(optimized_point)} bytes")
    
    # Demonstrate efficient data structures
    efficient_structures = optimizer.memory_efficient_data_structures()
    print(f"Regular list size: {efficient_structures['regular_list_size']} bytes")
    print(f"Efficient array size: {efficient_structures['efficient_array_size']} bytes")
    
    # Demonstrate string optimization
    string_opts = optimizer.optimize_string_operations()
    print(f"String interning works: {string_opts['interning_works']}")
    
    # Test string concatenation performance
    test_strings = [f"string_{i}" for i in range(1000)]
    
    start = time.time()
    result1 = string_opts['inefficient_concat'](test_strings)
    inefficient_time = time.time() - start
    
    start = time.time()
    result2 = string_opts['efficient_join'](test_strings)
    efficient_time = time.time() - start
    
    print(f"Inefficient concat time: {inefficient_time:.4f}s")
    print(f"Efficient join time: {efficient_time:.4f}s")
    print(f"Speedup: {inefficient_time / efficient_time:.2f}x")

def main():
    """Main demonstration function."""
    print("=== Advanced Python: Performance Optimization Demo ===")
    
    # Run individual demonstrations
    demonstrate_profiling()
    demonstrate_caching_strategies()
    demonstrate_memory_optimization()
    
    # Run comprehensive benchmark
    print("\n" + "="*60)
    benchmark = OptimizationBenchmark()
    summary = benchmark.run_full_benchmark()
    
    # Display benchmark results
    print("\n=== Benchmark Results Summary ===")
    
    if 'performance_improvements' in summary:
        improvements = summary['performance_improvements']
        for optimization, speedup in improvements.items():
            print(f"{optimization}: {speedup:.2f}x speedup")
    
    # Show top optimization opportunities
    profiler_summary = summary.get('profiler_summary', {})
    if 'slowest_function' in profiler_summary:
        slowest = profiler_summary['slowest_function']
        print(f"\nSlowest operation: {slowest[0]} ({slowest[1]['execution_time']:.4f}s)")
    
    if 'memory_heaviest' in profiler_summary:
        heaviest = profiler_summary['memory_heaviest']
        print(f"Memory heaviest operation: {heaviest[0]} ({heaviest[1]['memory_delta']:.2f}MB)")
    
    print("\n=== Optimization Recommendations ===")
    print("1. Use caching for expensive, repeated computations")
    print("2. Vectorize operations with NumPy when possible")
    print("3. Use parallel processing for CPU-intensive tasks")
    print("4. Optimize memory usage with __slots__ and efficient data structures")
    print("5. Choose appropriate algorithms and data structures")
    print("6. Profile your code to identify bottlenecks")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()
