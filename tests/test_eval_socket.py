from pathlib import Path

import eval_socket
import pytest


class _ReadyResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _LiveProcess:
    returncode = None

    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        pass


def test_start_fdp_server_passes_dp_and_wm_checkpoint_overrides(monkeypatch):
    command_calls = []

    def fake_popen(command, cwd):
        command_calls.append((command, cwd))
        return _LiveProcess()

    monkeypatch.setattr(eval_socket.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(eval_socket, "urlopen", lambda *args, **kwargs: _ReadyResponse())

    process = eval_socket._start_fdp_server(
        config="configs/inference/fdp_cad_wm.yaml",
        port=7076,
        python_command="conda run --no-capture-output -n fdp python",
        checkpoint="teacher.ckpt",
        wm_checkpoint="cad_wm.pt",
        backbone="dinov3",
        startup_timeout=1,
    )

    assert process is not None
    command, cwd = command_calls[0]
    assert command[:6] == [
        "conda", "run", "--no-capture-output", "-n", "fdp", "python",
    ]
    assert command[6:12] == [
        "fdp_server.py", "--config", "configs/inference/fdp_cad_wm.yaml",
        "--host", "127.0.0.1", "--port",
    ]
    assert command[12:] == [
        "7076", "--checkpoint", "teacher.ckpt",
        "--wm-checkpoint", "cad_wm.pt", "--backbone", "dinov3",
    ]
    assert Path(cwd) == Path(eval_socket.__file__).resolve().parent


def test_start_fdp_server_rejects_an_existing_listener(monkeypatch):
    class _ExistingSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        eval_socket.socket, "create_connection", lambda *args, **kwargs: _ExistingSocket()
    )
    with pytest.raises(RuntimeError, match="port 7075 is already in use"):
        eval_socket._start_fdp_server(
            config="configs/inference/fdp_cad_wm.yaml",
            port=7075,
            python_command="python",
        )
