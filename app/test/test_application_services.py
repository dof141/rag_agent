import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cryptography.fernet import Fernet

from app.application_services import (
    ApplicationConfigurationError,
    create_application_services,
)


class ApplicationServicesTest(unittest.TestCase):
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
            config = {
                "RAG_SQLITE_PATH": str(Path(temp_dir) / "rag.db"),
                "RAG_ADMIN_USERNAME": "admin",
                "RAG_ADMIN_PASSWORD": "first-password",
                "RAG_JWT_SECRET": "unit-test-signing-secret",
                "RAG_JWT_TTL_SECONDS": "60",
                "RAG_SETTINGS_MASTER_KEY": Fernet.generate_key().decode("ascii"),
            }
            services = create_application_services(config)
            services.initialize()

            changed = dict(config)
            changed["RAG_ADMIN_PASSWORD"] = "replacement"
            second = create_application_services(changed)
            second.initialize()

            self.assertIsNotNone(second.users.verify_credentials("admin", "first-password"))
            self.assertIsNone(second.users.verify_credentials("admin", "replacement"))


if __name__ == "__main__":
    unittest.main()
