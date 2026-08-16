import importlib
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


class ImportScriptIsolationTest(unittest.TestCase):
    def test_importing_manual_graph_script_does_not_execute_graph(self):
        target_module = "app.test.test_import_main_graph"

        class FailOnUseGraph:
            def stream(self, *args, **kwargs):
                raise AssertionError("导入测试模块时不应执行知识库导入图")

        main_graph = ModuleType("app.import_process.agent.main_graph")
        main_graph.kb_work_app = FailOnUseGraph()
        state = ModuleType("app.import_process.agent.state")
        state.create_default_state = Mock(return_value={})
        logger_module = ModuleType("app.core.logger")
        logger_module.logger = Mock()

        sys.modules.pop(target_module, None)
        with patch.dict(
            sys.modules,
            {
                "app.import_process.agent.main_graph": main_graph,
                "app.import_process.agent.state": state,
                "app.core.logger": logger_module,
            },
        ):
            module = importlib.import_module(target_module)

        self.assertTrue(callable(module.main))

if __name__ == "__main__":
    unittest.main()
