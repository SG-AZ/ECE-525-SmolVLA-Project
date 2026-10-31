"""checking the local dashboard api with real simulator observations."""

from gradio_client import Client

from .app import build_app
from .backend import Backend


def main() -> None:
    backend = Backend()
    app = build_app(backend)
    try:
        app.launch(
            server_name="127.0.0.1", server_port=7861, share=False, prevent_thread_lock=True, quiet=True
        )
        client = Client("http://127.0.0.1:7861", verbose=False)
        result = client.predict(0, next(iter(backend.targets)), api_name="/reset")
        assert result[0] and result[1], result
        assert len(result[2]) == 8, result
        assert result[-1].startswith("LIVE"), result[-1]
        print(
            "PASS: Gradio HTTP Start -> reset plus real LIBERO cameras, 8D state, instruction, status",
            flush=True,
        )
    finally:
        app.close()
        backend.close()


if __name__ == "__main__":
    main()
