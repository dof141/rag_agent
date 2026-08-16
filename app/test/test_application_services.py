import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from cryptography.fernet import Fernet

from app.application_services import (
    ApplicationConfigurationError,
    create_application_services,
    create_application_services_from_env,
    create_import_runtime,
)
from app.query_process.runtime import create_query_runtime


class ApplicationServicesTest(unittest.TestCase):
    def config(self, temp_dir):
        return {
            "RAG_SQLITE_PATH": str(Path(temp_dir) / "rag.db"),
            "RAG_ADMIN_USERNAME": "admin",
            "RAG_ADMIN_PASSWORD": "first-password",
            "RAG_JWT_SECRET": "unit-test-signing-secret",
            "RAG_JWT_TTL_SECONDS": "60",
            "RAG_SETTINGS_MASTER_KEY": Fernet.generate_key().decode("ascii"),
        }

    def test_missing_config_reports_names_without_values(self):
        with self.assertRaises(ApplicationConfigurationError) as raised:
            create_application_services(
                {
                    "RAG_SQLITE_PATH": "rag.db",
                    "RAG_ADMIN_USERNAME": "admin",
                }
            )

        message = str(raised.exception)
        self.assertIn("RAG_ADMIN_PASSWORD", message)
        self.assertIn("RAG_JWT_SECRET", message)
        self.assertIn("RAG_SETTINGS_MASTER_KEY", message)

    def test_initialize_creates_admin_once(self):
        with TemporaryDirectory() as temp_dir:
            config = self.config(temp_dir)
            services = create_application_services(config)
            services.initialize()

            changed = dict(config)
            changed["RAG_ADMIN_PASSWORD"] = "replacement"
            second = create_application_services(changed)
            second.initialize()

            self.assertIsNotNone(second.users.verify_credentials("admin", "first-password"))
            self.assertIsNone(second.users.verify_credentials("admin", "replacement"))

    def test_services_repr_does_not_include_admin_password(self):
        with TemporaryDirectory() as temp_dir:
            services = create_application_services(self.config(temp_dir))

        self.assertNotIn("first-password", repr(services))

    def test_output_root_is_loaded_from_environment(self):
        with TemporaryDirectory() as temp_dir:
            config = self.config(temp_dir)
            config["RAG_OUTPUT_ROOT"] = str(Path(temp_dir) / "imports")
            with patch.dict("os.environ", config, clear=True):
                services = create_application_services_from_env()

        self.assertEqual(services.output_root, Path(config["RAG_OUTPUT_ROOT"]))

    def test_services_bind_separate_import_and_query_runtime_factories(self):
        with TemporaryDirectory() as temp_dir:
            services = create_application_services(self.config(temp_dir))

        self.assertIs(services.import_runtime_factory, create_import_runtime)
        self.assertIs(services.query_runtime_factory, create_query_runtime)


if __name__ == "__main__":
    unittest.main()
