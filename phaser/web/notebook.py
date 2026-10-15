"""Jupyter integration: run the web server from a notebook and embed its views."""
import atexit
import threading
import multiprocessing
from multiprocessing.connection import Connection
import logging
import io
import os
import errno
import typing as t
import weakref
import yaml
try:
    from yaml import CSafeDumper as Dumper
except ImportError:
    from yaml import SafeDumper as Dumper

import requests
from IPython.display import display
import ipywidgets

if t.TYPE_CHECKING:
    from phaser.plan import ReconsPlan


def _random_port() -> int:
    import socket

    try:
        sock = socket.socket()
        sock.bind(('', 0))
        port = sock.getsockname()[1]
        sock.close()
    except OSError as e:
        raise RuntimeError("Couldn't find an open port") from e 
    return port


class _ServerLogHandler(logging.Handler):
    def __init__(self, conn: Connection):
        self.conn = conn
        super().__init__(logging.INFO)

    def emit(self, record: logging.LogRecord):
        try:
            self.conn.send(('log', self.format(record)))
        except Exception:
            self.handleError(record)


class _ServerConnection:
    def __init__(self, port: t.Optional[int],
                 output: t.Optional[ipywidgets.Output] = None):
        self.output: t.Optional[ipywidgets.Output] = output

        (self.conn, child_conn) = multiprocessing.Pipe()

        self._proc = multiprocessing.Process(
            target=self._run_server, args=[child_conn, port],
            daemon=False,
        )
        self._proc.start()

        # attempt to stop process, even though it's not a daemon process
        weak = weakref.proxy(self)
        self._atexit_handler = atexit.register(lambda: _ServerConnection._cleanup(weak))

        while True:
            msg: t.Tuple[str, t.Any] = self.conn.recv()
            if msg[0] == 'serving':
                self.port: int = msg[1]
                break
            elif msg[0] == 'log':
                if self.output:
                    self.output.append_stdout(msg[1] + '\n')
            elif msg[0] == 'exc':
                self._proc.join()
                raise msg[1]
            else:
                raise ValueError(f"Unknown message type '{msg[0]}'")

        self._monitoring_thread = threading.Thread(
            target=self._monitor_conn, daemon=True
        )
        self._monitoring_thread.start()

    @staticmethod
    def _cleanup(weak: t.Callable[[], t.Optional['_ServerConnection']]):
        if (self := weak()):
            self.stop()

    def stop(self):
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(5.0)
            if self._proc.is_alive():
                if self.output:
                    self.output.append_stderr("Failed to SIGTERM server, killing instead.\n")
                self._proc.kill()

            if self.output:
                self.output.append_stdout("Server stopped\n")

        if self._monitoring_thread.is_alive():
            self._monitoring_thread.join()

    def __del__(self):
        self.stop()
        atexit.unregister(self._atexit_handler)

    def is_alive(self):
        return self._proc.is_alive()

    def _monitor_conn(self):
        while True:
            msg: t.Tuple[str, t.Any] = self.conn.recv()
            if msg[0] == 'log':
                if self.output:
                    self.output.append_stdout(msg[1] + '\n')
            elif msg[0] == 'exc':
                if self.output:
                    self.output.append_stderr(msg[1] + '\n')
                break
            elif msg[0] == 'stop':
                break

    @staticmethod
    def _run_server(conn: Connection, port: t.Optional[int] = None):
        import sys
        from phaser.web.server import server

        sys.stdout = open(os.devnull)
        sys.stderr = open(os.devnull)

        logging.basicConfig(level=logging.INFO, handlers=[_ServerLogHandler(conn)])

        try:
            if port is not None:
                root_path = _ServerConnection.get_root_path(port)
                server.run(port=port, root_path=root_path,
                           serving_cb=lambda: conn.send(('serving', port)))
                return

            while True:
                port = _random_port()
                try:
                    root_path = _ServerConnection.get_root_path(port)
                    server.run(port=port, root_path=root_path,
                               serving_cb=lambda: conn.send(('serving', port)))
                    break
                except OSError as e:
                    if e.errno != errno.EADDRINUSE:
                        raise

        except KeyboardInterrupt:
            conn.send(('stop', None))
        except BaseException as e:
            conn.send(('exc', e))
        else:
            conn.send(('stop', None))

    @staticmethod
    def get_root_path(port: int) -> str:
        service_prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "/")
        return f"{service_prefix}proxy/absolute/{port}"

    @staticmethod
    def get_local_url(port: int) -> str:
        root_path = _ServerConnection.get_root_path(port)
        return f"http://localhost:{port}{root_path}"


def iframe(src: str, width: t.Any, height: t.Any) -> ipywidgets.HTML:
    """Return a widget embedding an iframe of `src`."""
    return ipywidgets.HTML(f'\
<iframe width="{width}" height="{height}" src="{src}" frameborder="0" allowfullscreen></iframe>\
')


class Manager:
    """Runs the web server in a subprocess and displays its views in a notebook.

    Args:
        port: Port to serve on. A random free port if `None`.
    """
    def __init__(self, port: t.Optional[int] = None):
        self.port: t.Optional[int] = port

        self._server: t.Optional[_ServerConnection] = None
        self.log_queue: multiprocessing.SimpleQueue = multiprocessing.SimpleQueue()
        self.out: t.Optional[ipywidgets.Output] = None

    def is_running(self) -> bool:
        """Return whether the server process is alive."""
        return self._server is not None and self._server.is_alive()

    def start(self):
        """Start the server and display its logs and job manager.

        Raises:
            RuntimeError: If the server is already running.
        """
        if self.is_running():
            raise RuntimeError("Manager already running")

        self.out = ipywidgets.Output()

        self._server = _ServerConnection(self.port, self.out)

        display(ipywidgets.Accordion(children=[
            self.out, self.manager_view()
        ], titles=['Server logs', 'Job manager'], selected_index=1))

    def stop(self):
        """Stop the server."""
        if self._server:
            self._server.stop()
            self.out = None

    def __del__(self):
        self.stop()

    """
    @property
    def root_path(self) -> str:
        assert self._server is not None
        return f"{self.service_prefix}proxy/absolute/{self._server.port}"

    @property
    def base_url(self) -> str:
        assert self.port is not None
        return f"http://localhost:{self.port}{self.root_path}"
    """

    def start_job(self, plan: 'ReconsPlan'):
        """Submit `plan` (a `ReconsPlan` or its YAML) to the server and display the job view.

        Raises:
            ValueError: If the server is not running or rejects the plan.
        """
        from phaser.plan import ReconsPlan

        if not self.is_running():
            raise ValueError("Server is not running")
        assert self._server is not None

        if not isinstance(plan, ReconsPlan):
            plan = ReconsPlan.from_yaml(plan)

        buf = io.StringIO()
        yaml.dump(plan.into_data(), buf, Dumper)

        base_url = self._server.get_local_url(self._server.port)

        resp = requests.post(
            f"{base_url}/job/start",
            json={
                'source': 'yaml',
                'data': buf.getvalue(),
            }
        )
        resp.raise_for_status()
        result = resp.json()
        if result['result'] == 'error':
            raise ValueError(result['msg'])

        jobs = result['jobs']
        assert len(jobs) == 1
        display(self.job_view(jobs[0]['job_id']))

    def manager_view(self) -> ipywidgets.HTML:
        """Return a widget showing the job manager.

        Raises:
            ValueError: If the server is not running.
        """
        if not self.is_running():
            raise ValueError("Server is not running")
        assert self._server is not None
        root_path = self._server.get_root_path(self._server.port)
        return iframe(
            f'{root_path}/', width=800, height=500
        )

    def job_view(self, job_id: str) -> ipywidgets.HTML:
        """Return a widget showing the dashboard for job `job_id`.

        Raises:
            ValueError: If the server is not running.
        """
        if not self.is_running():
            raise ValueError("Server is not running")
        assert self._server is not None
        root_path = self._server.get_root_path(self._server.port)
        return iframe(
            f'{root_path}/job/{job_id}',
            width=1200, height=2000
        )
