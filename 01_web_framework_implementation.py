"""
Advanced Python: Custom Web Framework Implementation
This script demonstrates building a lightweight web framework from scratch,
showcasing advanced Python concepts like metaclasses, decorators, and WSGI.
"""

import re
import json
import urllib.parse
from typing import Dict, List, Callable, Any, Optional, Tuple
from functools import wraps
from collections import defaultdict
import threading
import time
from wsgiref.simple_server import make_server

# Custom exceptions
class FrameworkError(Exception):
    """Base exception for framework errors."""
    pass

class RouteNotFound(FrameworkError):
    """Exception raised when no route matches the request."""
    pass

class MiddlewareError(FrameworkError):
    """Exception raised in middleware processing."""
    pass

# Request and Response classes
class Request:
    """HTTP Request wrapper."""
    
    def __init__(self, environ: Dict[str, Any]):
        self.environ = environ
        self.method = environ.get('REQUEST_METHOD', 'GET')
        self.path = environ.get('PATH_INFO', '/')
        self.query_string = environ.get('QUERY_STRING', '')
        self.headers = self._parse_headers(environ)
        self.body = self._read_body(environ)
        self._json = None
        self._form = None
    
    def _parse_headers(self, environ: Dict[str, Any]) -> Dict[str, str]:
        """Parse HTTP headers from WSGI environ."""
        headers = {}
        for key, value in environ.items():
            if key.startswith('HTTP_'):
                header_name = key[5:].replace('_', '-').title()
                headers[header_name] = value
        return headers
    
    def _read_body(self, environ: Dict[str, Any]) -> bytes:
        """Read request body."""
        try:
            content_length = int(environ.get('CONTENT_LENGTH', 0))
        except ValueError:
            content_length = 0
        
        if content_length > 0:
            return environ['wsgi.input'].read(content_length)
        return b''
    
    @property
    def json(self) -> Optional[Dict[str, Any]]:
        """Parse JSON body."""
        if self._json is None and self.body:
            try:
                self._json = json.loads(self.body.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._json = None
        return self._json
    
    @property
    def form(self) -> Dict[str, str]:
        """Parse form data."""
        if self._form is None:
            if self.headers.get('Content-Type', '').startswith('application/x-www-form-urlencoded'):
                self._form = dict(urllib.parse.parse_qsl(self.body.decode('utf-8')))
            else:
                self._form = {}
        return self._form
    
    @property
    def args(self) -> Dict[str, str]:
        """Parse query parameters."""
        return dict(urllib.parse.parse_qsl(self.query_string))

class Response:
    """HTTP Response wrapper."""
    
    def __init__(self, body: str = '', status: int = 200, headers: Optional[Dict[str, str]] = None):
        self.body = body
        self.status = status
        self.headers = headers or {}
        self.headers.setdefault('Content-Type', 'text/html; charset=utf-8')
    
    def json(self, data: Any) -> 'Response':
        """Set JSON response."""
        self.body = json.dumps(data)
        self.headers['Content-Type'] = 'application/json'
        return self
    
    def set_cookie(self, name: str, value: str, max_age: Optional[int] = None) -> 'Response':
        """Set a cookie."""
        cookie = f"{name}={value}"
        if max_age:
            cookie += f"; Max-Age={max_age}"
        self.headers['Set-Cookie'] = cookie
        return self

# Route handling
class Route:
    """Represents a single route."""
    
    def __init__(self, pattern: str, handler: Callable, methods: List[str]):
        self.pattern = pattern
        self.handler = handler
        self.methods = [m.upper() for m in methods]
        self.regex = self._compile_pattern(pattern)
    
    def _compile_pattern(self, pattern: str) -> re.Pattern:
        """Compile route pattern to regex."""
        # Convert Flask-style routes to regex
        pattern = re.sub(r'<(\w+)>', r'(?P<\1>[^/]+)', pattern)
        pattern = f'^{pattern}$'
        return re.compile(pattern)
    
    def match(self, path: str, method: str) -> Optional[Dict[str, str]]:
        """Check if route matches path and method."""
        if method.upper() not in self.methods:
            return None
        
        match = self.regex.match(path)
        if match:
            return match.groupdict()
        return None

# Middleware system
class Middleware:
    """Base middleware class."""
    
    def process_request(self, request: Request) -> Optional[Response]:
        """Process request before routing."""
        return None
    
    def process_response(self, request: Request, response: Response) -> Response:
        """Process response after handler."""
        return response

class LoggingMiddleware(Middleware):
    """Middleware for request logging."""
    
    def process_request(self, request: Request) -> Optional[Response]:
        start_time = time.time()
        request._start_time = start_time
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {request.method} {request.path}")
        return None
    
    def process_response(self, request: Request, response: Response) -> Response:
        if hasattr(request, '_start_time'):
            duration = time.time() - request._start_time
            print(f"Response: {response.status} ({duration:.3f}s)")
        return response

class CORSMiddleware(Middleware):
    """Middleware for CORS headers."""
    
    def __init__(self, allowed_origins: List[str] = None):
        self.allowed_origins = allowed_origins or ['*']
    
    def process_response(self, request: Request, response: Response) -> Response:
        response.headers['Access-Control-Allow-Origin'] = ', '.join(self.allowed_origins)
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

# Dependency injection system
class DependencyContainer:
    """Simple dependency injection container."""
    
    def __init__(self):
        self._services = {}
        self._singletons = {}
    
    def register(self, name: str, factory: Callable, singleton: bool = False):
        """Register a service."""
        self._services[name] = (factory, singleton)
    
    def get(self, name: str) -> Any:
        """Get a service instance."""
        if name not in self._services:
            raise ValueError(f"Service '{name}' not registered")
        
        factory, is_singleton = self._services[name]
        
        if is_singleton:
            if name not in self._singletons:
                self._singletons[name] = factory()
            return self._singletons[name]
        
        return factory()

# Template engine
class TemplateEngine:
    """Simple template engine."""
    
    def __init__(self):
        self.templates = {}
    
    def register_template(self, name: str, content: str):
        """Register a template."""
        self.templates[name] = content
    
    def render(self, name: str, context: Dict[str, Any] = None) -> str:
        """Render a template with context."""
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        
        template = self.templates[name]
        context = context or {}
        
        # Simple variable substitution
        for key, value in context.items():
            template = template.replace(f'{{{{{key}}}}}', str(value))
        
        return template

# Main framework class
class WebFramework:
    """Lightweight web framework."""
    
    def __init__(self):
        self.routes: List[Route] = []
        self.middleware: List[Middleware] = []
        self.error_handlers: Dict[int, Callable] = {}
        self.before_request_handlers: List[Callable] = []
        self.after_request_handlers: List[Callable] = []
        self.container = DependencyContainer()
        self.template_engine = TemplateEngine()
        self._setup_default_error_handlers()
    
    def _setup_default_error_handlers(self):
        """Setup default error handlers."""
        self.error_handlers[404] = lambda req: Response("Not Found", 404)
        self.error_handlers[500] = lambda req, exc: Response("Internal Server Error", 500)
    
    def route(self, path: str, methods: List[str] = None):
        """Decorator for registering routes."""
        if methods is None:
            methods = ['GET']
        
        def decorator(handler):
            self.add_route(path, handler, methods)
            return handler
        
        return decorator
    
    def add_route(self, path: str, handler: Callable, methods: List[str]):
        """Add a route to the framework."""
        route = Route(path, handler, methods)
        self.routes.append(route)
    
    def add_middleware(self, middleware: Middleware):
        """Add middleware to the framework."""
        self.middleware.append(middleware)
    
    def before_request(self, handler: Callable):
        """Register before request handler."""
        self.before_request_handlers.append(handler)
        return handler
    
    def after_request(self, handler: Callable):
        """Register after request handler."""
        self.after_request_handlers.append(handler)
        return handler
    
    def error_handler(self, status_code: int):
        """Decorator for registering error handlers."""
        def decorator(handler):
            self.error_handlers[status_code] = handler
            return handler
        return decorator
    
    def _find_route(self, path: str, method: str) -> Tuple[Route, Dict[str, str]]:
        """Find matching route for path and method."""
        for route in self.routes:
            params = route.match(path, method)
            if params is not None:
                return route, params
        raise RouteNotFound(f"No route found for {method} {path}")
    
    def _process_middleware_request(self, request: Request) -> Optional[Response]:
        """Process request through middleware."""
        for middleware in self.middleware:
            response = middleware.process_request(request)
            if response:
                return response
        return None
    
    def _process_middleware_response(self, request: Request, response: Response) -> Response:
        """Process response through middleware."""
        for middleware in reversed(self.middleware):
            response = middleware.process_response(request, response)
        return response
    
    def _inject_dependencies(self, handler: Callable, request: Request, params: Dict[str, str]) -> Dict[str, Any]:
        """Inject dependencies into handler."""
        import inspect
        
        sig = inspect.signature(handler)
        kwargs = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'request':
                kwargs[param_name] = request
            elif param_name in params:
                kwargs[param_name] = params[param_name]
            elif hasattr(param.annotation, '__name__'):
                # Try to resolve from container
                try:
                    kwargs[param_name] = self.container.get(param.annotation.__name__)
                except ValueError:
                    pass
        
        return kwargs
    
    def handle_request(self, request: Request) -> Response:
        """Handle a single request."""
        try:
            # Process middleware
            middleware_response = self._process_middleware_request(request)
            if middleware_response:
                return self._process_middleware_response(request, middleware_response)
            
            # Run before request handlers
            for handler in self.before_request_handlers:
                handler(request)
            
            # Find and execute route
            route, params = self._find_route(request.path, request.method)
            
            # Inject dependencies
            kwargs = self._inject_dependencies(route.handler, request, params)
            
            # Call handler
            result = route.handler(**kwargs)
            
            # Convert result to Response
            if isinstance(result, Response):
                response = result
            elif isinstance(result, str):
                response = Response(result)
            elif isinstance(result, dict):
                response = Response().json(result)
            else:
                response = Response(str(result))
            
            # Run after request handlers
            for handler in self.after_request_handlers:
                response = handler(request, response) or response
            
            return self._process_middleware_response(request, response)
        
        except RouteNotFound:
            return self._handle_error(request, 404)
        except Exception as e:
            print(f"Error handling request: {e}")
            return self._handle_error(request, 500, e)
    
    def _handle_error(self, request: Request, status_code: int, exception: Exception = None) -> Response:
        """Handle errors using registered error handlers."""
        if status_code in self.error_handlers:
            handler = self.error_handlers[status_code]
            try:
                if exception:
                    return handler(request, exception)
                else:
                    return handler(request)
            except Exception:
                pass
        
        return Response(f"Error {status_code}", status_code)
    
    def wsgi_app(self, environ: Dict[str, Any], start_response: Callable):
        """WSGI application interface."""
        request = Request(environ)
        response = self.handle_request(request)
        
        # Prepare WSGI response
        status = f"{response.status} {self._get_status_text(response.status)}"
        headers = list(response.headers.items())
        
        start_response(status, headers)
        return [response.body.encode('utf-8')]
    
    def _get_status_text(self, status_code: int) -> str:
        """Get status text for status code."""
        status_texts = {
            200: 'OK',
            201: 'Created',
            400: 'Bad Request',
            401: 'Unauthorized',
            403: 'Forbidden',
            404: 'Not Found',
            500: 'Internal Server Error'
        }
        return status_texts.get(status_code, 'Unknown')
    
    def run(self, host: str = 'localhost', port: int = 8000, debug: bool = False):
        """Run the development server."""
        print(f"Starting server on http://{host}:{port}")
        if debug:
            print("Debug mode enabled")
        
        server = make_server(host, port, self.wsgi_app)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")

# Example application
def create_example_app() -> WebFramework:
    """Create an example application."""
    app = WebFramework()
    
    # Add middleware
    app.add_middleware(LoggingMiddleware())
    app.add_middleware(CORSMiddleware())
    
    # Register templates
    app.template_engine.register_template('home', '''
    <html>
        <head><title>{{title}}</title></head>
        <body>
            <h1>{{heading}}</h1>
            <p>{{message}}</p>
        </body>
    </html>
    ''')
    
    # Register services
    class DatabaseService:
        def __init__(self):
            self.data = {'users': [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}]}
        
        def get_users(self):
            return self.data['users']
    
    app.container.register('DatabaseService', DatabaseService, singleton=True)
    
    # Routes
    @app.route('/')
    def home(request: Request):
        return app.template_engine.render('home', {
            'title': 'My Web Framework',
            'heading': 'Welcome!',
            'message': 'This is a custom Python web framework.'
        })
    
    @app.route('/api/users')
    def get_users(request: Request, DatabaseService: DatabaseService):
        users = DatabaseService.get_users()
        return {'users': users}
    
    @app.route('/api/users/<user_id>')
    def get_user(request: Request, user_id: str, DatabaseService: DatabaseService):
        users = DatabaseService.get_users()
        user = next((u for u in users if str(u['id']) == user_id), None)
        if user:
            return {'user': user}
        return Response('User not found', 404)
    
    @app.route('/api/echo', methods=['POST'])
    def echo(request: Request):
        if request.json:
            return {'echo': request.json}
        return {'echo': request.form}
    
    @app.before_request
    def log_request(request: Request):
        print(f"Processing {request.method} request to {request.path}")
    
    @app.after_request
    def add_headers(request: Request, response: Response):
        response.headers['X-Framework'] = 'CustomPython'
        return response
    
    @app.error_handler(404)
    def not_found(request: Request):
        return Response('Page not found!', 404)
    
    return app

def demonstrate_framework():
    """Demonstrate the web framework."""
    print("=== Custom Web Framework Demo ===")
    
    app = create_example_app()
    
    # Simulate requests for demonstration
    from wsgiref.util import setup_testing_defaults
    
    def simulate_request(method: str, path: str, body: str = ''):
        environ = {
            'REQUEST_METHOD': method,
            'PATH_INFO': path,
            'QUERY_STRING': '',
            'CONTENT_LENGTH': str(len(body)),
            'wsgi.input': type('MockInput', (), {'read': lambda self, size: body.encode()})()
        }
        setup_testing_defaults(environ)
        
        request = Request(environ)
        response = app.handle_request(request)
        
        print(f"\n{method} {path}")
        print(f"Status: {response.status}")
        print(f"Body: {response.body[:100]}...")
        return response
    
    # Test routes
    simulate_request('GET', '/')
    simulate_request('GET', '/api/users')
    simulate_request('GET', '/api/users/1')
    simulate_request('GET', '/nonexistent')
    simulate_request('POST', '/api/echo', '{"message": "Hello World"}')

if __name__ == "__main__":
    demonstrate_framework()
    
    # Uncomment to run the server
    # app = create_example_app()
    # app.run(debug=True)
