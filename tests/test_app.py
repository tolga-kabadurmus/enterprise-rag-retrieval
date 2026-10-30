"""Tests for the initial FastAPI scaffold."""

import unittest

from fastapi import FastAPI

from app.main import app, create_app


class ApplicationScaffoldTests(unittest.TestCase):
    """Verify that the ASGI application can be imported and created."""

    def test_module_exposes_fastapi_application(self) -> None:
        self.assertIsInstance(app, FastAPI)

    def test_application_factory_sets_service_identity(self) -> None:
        application = create_app()
        self.assertEqual(application.title, "Enterprise RAG Query")
        self.assertEqual(application.version, "0.1.0")


if __name__ == "__main__":
    unittest.main()
