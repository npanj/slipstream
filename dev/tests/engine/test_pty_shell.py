import unittest
from server.agent.pty_shell import WarmPtyShell


class WarmPtyShellTest(unittest.TestCase):
    def test_run_command_execution_and_exit_code(self):
        shell = WarmPtyShell()
        try:
            # 1. Simple command execution
            code, output = shell.run_command("echo hello_world", timeout=5.0)
            self.assertEqual(code, 0)
            self.assertIn("hello_world", output)

            # 2. Environment state persistence across turns
            code, _ = shell.run_command("export SLIPSTREAM_AGENT_ENV=active", timeout=5.0)
            self.assertEqual(code, 0)

            code, output = shell.run_command("echo $SLIPSTREAM_AGENT_ENV", timeout=5.0)
            self.assertEqual(code, 0)
            self.assertIn("active", output)

            # 3. Non-zero exit code capture
            code, _ = shell.run_command("false", timeout=5.0)
            self.assertEqual(code, 1)
        finally:
            shell.close()


if __name__ == "__main__":
    unittest.main()
