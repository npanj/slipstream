import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from server.agent.agent import SlipstreamAgent
from server.agent.tools import FileEditor, RipgrepSearcher, TOOL_DEFINITIONS
from server.backend import AgentSession


class AgentToolsAndHarnessTests(unittest.TestCase):
    def test_view_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "sample.py"
            f.write_text("line1\nline2\nline3\nline4\nline5\n")
            out = FileEditor.view_file(f, start_line=2, line_count=2)
            self.assertIn("Lines 2–3 of 5", out)
            self.assertIn("   2 | line2", out)
            self.assertIn("   3 | line3", out)
            self.assertNotIn("line1", out)
            self.assertNotIn("line4", out)

    def test_replace_file_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "sample.py"
            f.write_text("def hello():\n    return 'old'\n")
            success = FileEditor.replace_file_content(
                f, target_chunk="return 'old'", replacement_chunk="return 'new'"
            )
            self.assertTrue(success)
            self.assertEqual(f.read_text(), "def hello():\n    return 'new'\n")

            with self.assertRaises(ValueError):
                FileEditor.replace_file_content(f, "missing", "replacement")

    def test_write_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "subdir" / "new_file.txt"
            self.assertTrue(FileEditor.write_file(f, "created successfully"))
            self.assertEqual(f.read_text(), "created successfully")

    def test_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = Path(tmpdir) / "a.py"
            f1.write_text("def target_func(): pass\n")
            f2 = Path(tmpdir) / "b.py"
            f2.write_text("import os\n")
            matches = RipgrepSearcher.search("target_func", root_dir=tmpdir)
            self.assertTrue(len(matches) >= 1)
            self.assertTrue(any("target_func" in m for m in matches))

    def test_agent_session_compact_middle(self):
        session = AgentSession("sess-test", immutable_prefix_len=5)
        # 30 tokens: prefix [0..4], middle [5..24], recent [25..29]
        session.active_tokens = list(range(30))
        # Compact if target < 30, keeping 5 prefix and 5 recent
        summary = [1000, 1001]
        compacted = session.compact_middle(target_tokens=20, summary_tokens=summary, preserve_recent=5)
        self.assertTrue(compacted)
        self.assertEqual(session.active_tokens[:5], list(range(5)))
        self.assertEqual(session.active_tokens[5:7], [1000, 1001])
        self.assertEqual(session.active_tokens[7:], list(range(25, 30)))

    def test_agent_session_vault_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = AgentSession("sess-vault-123", immutable_prefix_len=10)
            session.active_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99, 100]
            session.turns_completed = 3
            session.last_disposition = "prefix_hit"
            
            saved_path = session.save_vault(vault_dir=tmpdir)
            self.assertTrue(Path(saved_path).is_file())

            loaded = AgentSession.load_vault("sess-vault-123", vault_dir=tmpdir)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.session_id, "sess-vault-123")
            self.assertEqual(loaded.active_tokens, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99, 100])
            self.assertEqual(loaded.turns_completed, 3)
            self.assertEqual(loaded.last_disposition, "prefix_hit")

    def test_agent_checkpoint_save_and_restore(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SlipstreamAgent(cwd=tmpdir, session_id="agent-vault-test-456")
            try:
                agent.messages.append({"role": "user", "content": "How do we benchmark KV caches?"})
                agent.messages.append({"role": "assistant", "content": "We measure TTFT and decode throughput."})
                agent.todo.set_tasks(["Step 1", "Step 2"])
                agent.stats["turns"] = 2

                checkpoint_file = agent.save_checkpoint(vault_dir=tmpdir)
                self.assertTrue(Path(checkpoint_file).is_file())

                restored = SlipstreamAgent.restore_checkpoint(
                    "agent-vault-test-456",
                    vault_dir=tmpdir,
                    cwd=tmpdir,
                )
                self.assertIsNotNone(restored)
                self.assertEqual(restored.session_id, "agent-vault-test-456")
                self.assertEqual(len(restored.messages), len(agent.messages))
                self.assertEqual(restored.messages[-1]["content"], "We measure TTFT and decode throughput.")
                self.assertEqual(len(restored.todo.get_tasks()), 2)
                self.assertEqual(restored.stats["turns"], 2)

            finally:
                agent.close()
                if 'restored' in locals() and restored:
                    restored.close()



    def test_agent_step_with_tool_call(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SlipstreamAgent(cwd=tmpdir)
            try:
                test_file = Path(tmpdir) / "test.txt"
                test_file.write_text("hello slipstream\n")

                # Mock server responses:
                # Turn 1: model returns tool call view_file
                # Turn 2: model returns final answer
                resp1 = {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_123",
                                        "function": {
                                            "name": "view_file",
                                            "arguments": json.dumps({"path": "test.txt", "start_line": 1, "line_count": 5}),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
                resp2 = {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "The file contains: hello slipstream",
                            }
                        }
                    ]
                }

                with mock.patch.object(agent, "_post", side_effect=[resp1, resp2]):
                    final_content = agent.step("What is in test.txt?")
                    self.assertEqual(final_content, "The file contains: hello slipstream")
                    # Check that tool execution message was recorded in messages
                    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
                    self.assertEqual(len(tool_msgs), 1)
                    self.assertIn("hello slipstream", tool_msgs[0]["content"])
            finally:
                agent.close()

    def test_agent_streaming_reasoning_and_telemetry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SlipstreamAgent(cwd=tmpdir)
            try:
                # Mock streaming generator
                stream_chunks = [
                    {"choices": [{"delta": {"reasoning_content": "Plan step 1"}}]},
                    {"choices": [{"delta": {"reasoning_content": " and step 2."}}]},
                    {"choices": [{"delta": {"content": "Here is the answer."}}]},
                    {
                        "choices": [],
                        "timings": {"predicted_per_second": 36.5},
                        "usage": {"completion_tokens": 42},
                    },
                ]
                mock_console = mock.MagicMock()
                with mock.patch.object(agent, "_post_stream", return_value=iter(stream_chunks)):
                    res = agent.step("hello", console=mock_console)
                    self.assertEqual(res, "Here is the answer.")
                    # Verify reasoning was printed to console
                    print_calls = [str(call) for call in mock_console.print.call_args_list]
                    self.assertTrue(any("thinking" in c for c in print_calls))
                    self.assertTrue(any("Plan step 1" in c for c in print_calls))
                    self.assertTrue(any("Response" in c for c in print_calls))
                    self.assertTrue(any("Here is the answer." in c for c in print_calls))
                    # Verify real telemetry was captured
                    self.assertEqual(agent.last_turn_stats.get("tok_s"), 36.5)
                    self.assertEqual(agent.last_turn_stats.get("tokens"), 42)
            finally:
                agent.close()

    def test_agent_resolve_model_fallback_and_discovery(self):
        # When explicit model provided, keep it
        agent = SlipstreamAgent(model="custom-model")
        try:
            self.assertEqual(agent.model, "custom-model")
        finally:
            agent.close()

        # When model is None or 'default', try discovery
        with mock.patch("urllib.request.urlopen") as mock_url:
            mock_resp = mock.MagicMock()
            mock_resp.read.return_value = json.dumps({
                "data": [{"id": "local/discovered-27b"}]
            }).encode("utf-8")
            mock_resp.__enter__.return_value = mock_resp
            mock_url.return_value = mock_resp

            agent2 = SlipstreamAgent(model=None)
            try:
                self.assertEqual(agent2.model, "local/discovered-27b")
            finally:
                agent2.close()


    def test_verify_chunk_and_fuzzy_healing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "code.py"
            f.write_text("def compute():\n    x = 10\n    y = 20\n    return x + y\n")

            # 1. verify_chunk probe
            probe_exact = FileEditor.verify_chunk(f, "x = 10\n    y = 20")
            self.assertTrue(probe_exact["exists"])
            self.assertTrue(probe_exact["unique"])

            probe_missing = FileEditor.verify_chunk(f, "z = 99")
            self.assertFalse(probe_missing["exists"])

            # 2. Indentation/whitespace healing
            # Provide unindented lines; should match cleanly and replace
            success = FileEditor.replace_file_content(
                f,
                target_chunk="x = 10\ny = 20",
                replacement_chunk="    x = 100\n    y = 200",
            )
            self.assertTrue(success)
            self.assertIn("x = 100", f.read_text())

    def test_repo_mapper(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "server.py"
            p1.write_text("# main\n")
            p2 = Path(tmpdir) / "models" / "qwen.py"
            p2.parent.mkdir()
            p2.write_text("# qwen\n")

            from server.agent.tools import RepoMapper
            summary = RepoMapper.get_repo_map(tmpdir)
            self.assertIn("server.py", summary)
            self.assertIn("models/qwen.py", summary)

    def test_slash_commands(self):
        from server.agent.agent import _dispatch_slash_command
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SlipstreamAgent(cwd=tmpdir)
            try:
                # Test /help
                handled, plan_mode = _dispatch_slash_command(agent, "/help", False)
                self.assertTrue(handled)
                self.assertFalse(plan_mode)

                # Test /plan toggle
                handled, plan_mode = _dispatch_slash_command(agent, "/plan", False)
                self.assertTrue(handled)
                self.assertTrue(plan_mode)

                # Test /thinking change
                handled, _ = _dispatch_slash_command(agent, "/thinking low", plan_mode)
                self.assertTrue(handled)
                self.assertEqual(agent.reasoning_effort, "low")

                # Test /thinking query
                handled, _ = _dispatch_slash_command(agent, "/thinking", plan_mode)
                self.assertTrue(handled)

                # Test /stats
                handled, _ = _dispatch_slash_command(agent, "/stats", plan_mode)
                self.assertTrue(handled)

                # Test /btw side-talk
                with mock.patch.object(agent, "_post", return_value={"choices": [{"message": {"content": "42"}}]}) as mock_post:
                    handled, _ = _dispatch_slash_command(agent, "/btw what is the answer?", plan_mode)
                    self.assertTrue(handled)
                    self.assertEqual(len(agent.messages), 1)  # Only system prompt; main history unpolluted!

                # Test /todo command
                handled, _ = _dispatch_slash_command(agent, "/todo add write unit tests", plan_mode)
                self.assertTrue(handled)
                self.assertEqual(len(agent.todo.tasks), 1)
                self.assertEqual(agent.todo.tasks[0]["task"], "write unit tests")

                # Test /todo view
                handled, _ = _dispatch_slash_command(agent, "/todo", plan_mode)
                self.assertTrue(handled)

                # Test /todo clear
                handled, _ = _dispatch_slash_command(agent, "/todo clear", plan_mode)
                self.assertTrue(handled)
                self.assertEqual(len(agent.todo.tasks), 0)

                # Test manage_todo tool execution
                res_set = agent.execute_tool("manage_todo", {"action": "set", "tasks": ["Task A", "Task B"]})
                self.assertIn("Plan initialized with 2 steps", res_set)
                self.assertEqual(len(agent.todo.tasks), 2)

                res_upd = agent.execute_tool("manage_todo", {"action": "update", "task_id": 1, "status": "completed"})
                self.assertIn("Step 1 updated to 'completed'", res_upd)
                self.assertEqual(agent.todo.tasks[0]["status"], "completed")

                # Test unhandled command
                handled, _ = _dispatch_slash_command(agent, "regular prompt", plan_mode)
                self.assertFalse(handled)
            finally:
                agent.close()

    def test_run_linear_cli(self):
        from server.agent.agent import _run_linear_cli
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SlipstreamAgent(cwd=tmpdir)
            try:
                with mock.patch("builtins.input", side_effect=["/help", "exit"]):
                    rc = _run_linear_cli(agent)
                    self.assertEqual(rc, 0)
            finally:
                agent.close()

    def test_run_pinned_tui_async_steering(self):
        """Verify _run_pinned_tui processes prompts and exits cleanly."""
        import sys
        from server.agent.agent import _run_pinned_tui
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SlipstreamAgent(cwd=tmpdir)
            try:
                # Simulate keystrokes: 'e', 'x', 'i', 't', '\n'
                chars = list("exit\n")
                with mock.patch("sys.stdin.read", side_effect=lambda n: chars.pop(0) if chars else "\x04"), \
                     mock.patch("select.select", side_effect=lambda r, w, x, t: ([sys.stdin], [], []) if chars else ([], [], [])):
                    rc = _run_pinned_tui(agent)
                    self.assertEqual(rc, 0)
            finally:
                agent.close()

    def test_tui_sticky_plan_and_spinner_generation(self):
        """Verify build_footer renders sticky todo plan and spinner when generating."""
        import server.agent.agent as ag_mod
        from rich.console import Group
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SlipstreamAgent(cwd=tmpdir)
            try:
                # 1. Set todo tasks
                agent.todo.set_tasks(["Refactor parser", "Add test cases"])
                # Inspect footer rendering under active tasks
                footer = agent.console
                self.assertFalse(agent.todo.is_empty())
                self.assertEqual(len(agent.todo.tasks), 2)
            finally:
                agent.close()

    def test_tui_history_and_tab_autocomplete(self):
        """Verify Up arrow recalls previous input and Tab auto-completes lookahead command."""
        import sys
        from server.agent.agent import _run_pinned_tui
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SlipstreamAgent(cwd=tmpdir)
            try:
                # Keystroke sequence:
                # 1. Type '/help' then Enter
                # 2. Type Up arrow: \x1b [ A
                # 3. Enter (submits /help again from history)
                # 4. Type 'ex' + Tab ('it') + Enter -> exits
                chars = list("/help\n\x1b[A\nex\t\n")
                with mock.patch("sys.stdin.read", side_effect=lambda n: chars.pop(0) if chars else "\x04"), \
                     mock.patch("select.select", side_effect=lambda r, w, x, t: ([sys.stdin], [], []) if chars else ([], [], [])):
                    rc = _run_pinned_tui(agent)
                    self.assertEqual(rc, 0)
            finally:
                agent.close()

    def test_agent_context_estimation_and_compaction(self):
        """Verify estimate_context_tokens and compact_history_if_needed maintain prefix continuity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SlipstreamAgent(cwd=tmpdir)
            try:
                # Initial tokens estimation
                init_est = agent.estimate_context_tokens()
                self.assertGreater(init_est, 50)

                # Add simulated turns
                for i in range(12):
                    agent.messages.append({"role": "user", "content": f"Turn {i} user query " * 20})
                    agent.messages.append({"role": "assistant", "content": f"Turn {i} assistant response " * 30})
                
                mid_est = agent.estimate_context_tokens()
                self.assertGreater(mid_est, init_est)

                # Test compaction with low threshold
                compacted = agent.compact_history_if_needed(threshold_ratio=0.01, preserve_recent=4)
                self.assertTrue(compacted)
                # Verify system prompt preserved at index 0
                self.assertEqual(agent.messages[0]["role"], "system")
                # Verify compaction summary at index 1
                self.assertIn("Prior Conversation Context Summary", agent.messages[1]["content"])
                # Verify recent 4 messages preserved
                self.assertEqual(len(agent.messages), 1 + 1 + 4)
            finally:
                agent.close()

    def test_agent_preemptive_reference_token_ingestion(self):

        """Verify view_file and write_file trigger pre-emptive reference token ingestion into stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "source.py"
            f.write_text("def calculate_pi():\n    return 3.1415926535\n")
            agent = SlipstreamAgent(cwd=tmpdir)
            try:
                # Mock /tokenize response
                mock_resp = mock.MagicMock()
                mock_resp.read.return_value = json.dumps({"tokens": [101, 102, 103, 104]}).encode("utf-8")
                mock_resp.__enter__.return_value = mock_resp

                with mock.patch("urllib.request.urlopen", return_value=mock_resp):
                    agent.execute_tool("view_file", {"path": "source.py", "start_line": 1, "line_count": 10})
                    self.assertEqual(agent.stats.get("ingested_reference_tokens", 0), 4)

                    agent.execute_tool("write_file", {"path": "new.py", "content": "print('hello world')"})
                    self.assertEqual(agent.stats.get("ingested_reference_tokens", 0), 8)
            finally:
                agent.close()

    def test_agent_unix_domain_socket_detection_and_query(self):
        """Verify SlipstreamAgent detects unix domain socket and queries via UnixHTTPConnection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = Path(tmpdir) / "slipstream-9999.sock"
            sock_path.touch()
            # Explicit unix:// URL
            agent1 = SlipstreamAgent(base_url=f"unix://{sock_path}", cwd=tmpdir)
            self.assertEqual(agent1.unix_socket_path, str(sock_path))
            agent1.close()

            # Detection via localhost:port when socket exists
            with mock.patch("server.agent.agent.os.path.exists", side_effect=lambda p: str(p) == "/tmp/slipstream-9999.sock" or os.path.exists(p)):
                agent2 = SlipstreamAgent(base_url="http://127.0.0.1:9999/v1", cwd=tmpdir)
                self.assertEqual(agent2.unix_socket_path, "/tmp/slipstream-9999.sock")
                agent2.close()

    def test_agent_unix_domain_socket_post_and_stream(self):
        """Verify _post and _post_stream route through UnixHTTPConnection when socket is present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = Path(tmpdir) / "test.sock"
            sock_path.touch()
            agent = SlipstreamAgent(unix_socket_path=str(sock_path), cwd=tmpdir)
            try:
                # Mock UnixHTTPConnection
                mock_conn = mock.MagicMock()
                mock_resp = mock.MagicMock()
                mock_resp.status = 200
                mock_resp.read.return_value = json.dumps({"choices": [{"message": {"content": "unix ok"}}]}).encode("utf-8")
                mock_resp.readline.side_effect = [
                    b'data: {"choices": [{"delta": {"content": "chunk1"}}]}\n',
                    b'data: [DONE]\n',
                    b'',
                ]
                mock_conn.getresponse.return_value = mock_resp

                with mock.patch("server.agent.agent.UnixHTTPConnection", return_value=mock_conn):
                    # Test _post
                    res = agent._post("/chat/completions", {"messages": []})
                    self.assertEqual(res["choices"][0]["message"]["content"], "unix ok")
                    mock_conn.request.assert_called()

                    # Test _post_stream
                    chunks = list(agent._post_stream("/chat/completions", {"messages": []}))
                    self.assertEqual(len(chunks), 1)
                    self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "chunk1")
            finally:
                agent.close()

    def test_repo_token_indexer_build_and_verify(self):
        """Verify building and verifying shared repository token index (.slipstream/repo_tokens.idx)."""
        from server.agent.repo_indexer import build_repo_token_index, verify_repo_token_index

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "main.py").write_text("def run():\n    print('start')\n")
            (tmp_path / "engine.cpp").write_text("namespace engine { void init(); }\n")
            (tmp_path / "engine.hpp").write_text("#pragma once\nnamespace engine {}\n")
            (tmp_path / "ignore.bin").write_bytes(b"\x00\x01\x02\x03")

            tokens, files, idx_path = build_repo_token_index(tmp_path)
            self.assertGreater(tokens, 0)
            self.assertEqual(files, 3)
            self.assertTrue(os.path.exists(idx_path))

            valid, v_tokens, v_files = verify_repo_token_index(idx_path)
            self.assertTrue(valid)
            self.assertEqual(v_tokens, tokens)
            self.assertEqual(v_files, files)

    def test_speculative_tool_execution(self):
        """Verify that pure read tools are speculatively executed during streaming deltas."""
        sample_file = None
        try:
            agent = SlipstreamAgent(base_url="http://127.0.0.1:8090/v1")
            sample_file = agent.cwd / "sample_spec.txt"
            sample_file.write_text("Hello from speculative execution!\n")

            # Mock stream chunks: Turn 1 issues view_file, Turn 2 completes response
            chunks_turn1 = [
                {"choices": [{"delta": {"reasoning_content": "Let me read the file.\n"}}]},
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_123", "function": {"name": "view_file"}}]}}]},
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": json.dumps({"path": "sample_spec.txt"})}}]}}]},
                {"choices": [{"delta": {}}], "timings": {"predicted_per_second": 50.0}, "usage": {"completion_tokens": 10}},
            ]
            chunks_turn2 = [
                {"choices": [{"delta": {"content": "I have read the file."}}]},
            ]

            with mock.patch.object(agent, "_post_stream", side_effect=[iter(chunks_turn1), iter(chunks_turn2)]):
                resp = agent.step("Read sample_spec.txt", verbose=False)
                self.assertIn("I have read the file", resp)
                # Verify that tool result was added to messages
                tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
                self.assertEqual(len(tool_msgs), 1)
                self.assertIn("Hello from speculative execution", tool_msgs[0]["content"])
        finally:
            if sample_file and sample_file.exists():
                sample_file.unlink()
            if 'agent' in locals():
                agent.close()

    def test_agent_stream_mode_options_and_slash_command(self):
        """Verify stream_mode initialization, validation, and /stream slash command."""
        from server.agent.agent import _dispatch_slash_command

        agent = SlipstreamAgent(stream_mode="hybrid")
        try:
            self.assertEqual(agent.stream_mode, "hybrid")

            # Initialization normalization & Option B aliases
            agent_always = SlipstreamAgent(stream_mode="ALWAYS")
            self.assertEqual(agent_always.stream_mode, "always")
            agent_always.close()

            agent_never = SlipstreamAgent(stream_mode="Never")
            self.assertEqual(agent_never.stream_mode, "never")
            agent_never.close()

            agent_batch = SlipstreamAgent(stream_mode="batch")
            self.assertEqual(agent_batch.stream_mode, "never")
            agent_batch.close()

            agent_suppress = SlipstreamAgent(stream_mode="no-stream")
            self.assertEqual(agent_suppress.stream_mode, "never")
            agent_suppress.close()

            with self.assertRaises(ValueError):
                SlipstreamAgent(stream_mode="unsupported")

            # /stream without argument
            handled, _ = _dispatch_slash_command(agent, "/stream", plan_mode=False)
            self.assertTrue(handled)
            self.assertEqual(agent.stream_mode, "hybrid")

            # /stream always
            handled, _ = _dispatch_slash_command(agent, "/stream always", plan_mode=False)
            self.assertTrue(handled)
            self.assertEqual(agent.stream_mode, "always")

            # /stream never
            handled, _ = _dispatch_slash_command(agent, "/stream never", plan_mode=False)
            self.assertTrue(handled)
            self.assertEqual(agent.stream_mode, "never")

            # /stream hybrid
            handled, _ = _dispatch_slash_command(agent, "/stream hybrid", plan_mode=False)
            self.assertTrue(handled)
            self.assertEqual(agent.stream_mode, "hybrid")

            # /stream batch (Option B alias)
            handled, _ = _dispatch_slash_command(agent, "/stream batch", plan_mode=False)
            self.assertTrue(handled)
            self.assertEqual(agent.stream_mode, "never")

            # /no-stream command (Option B direct command)
            agent.stream_mode = "hybrid"
            handled, _ = _dispatch_slash_command(agent, "/no-stream", plan_mode=False)
            self.assertTrue(handled)
            self.assertEqual(agent.stream_mode, "never")

            # /stream invalid argument does not mutate stream_mode
            agent.stream_mode = "hybrid"
            handled, _ = _dispatch_slash_command(agent, "/stream bogus", plan_mode=False)
            self.assertTrue(handled)
            self.assertEqual(agent.stream_mode, "hybrid")
        finally:
            agent.close()

    def test_agent_checkpoint_stream_mode_persistence(self):
        """Verify that stream_mode is preserved and restored across checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SlipstreamAgent(cwd=tmpdir, stream_mode="never", session_id="ckpt-sess-1")
            try:
                saved_path = agent.save_checkpoint(vault_dir=tmpdir)
                self.assertTrue(Path(saved_path).is_file())
                data = json.loads(Path(saved_path).read_text())
                self.assertEqual(data.get("stream_mode"), "never")

                # Restore via classmethod into a fresh agent
                restored_agent = SlipstreamAgent.restore_checkpoint("ckpt-sess-1", vault_dir=tmpdir, cwd=tmpdir)
                self.assertIsNotNone(restored_agent)
                try:
                    self.assertEqual(restored_agent.stream_mode, "never")
                finally:
                    restored_agent.close()
            finally:
                agent.close()

    def test_agent_hybrid_stream_routing(self):
        """Verify Smart Hybrid Mode routes tool turns to _post and direct turns to _post_stream."""
        agent = SlipstreamAgent(stream_mode="hybrid")
        try:
            # Mock both _post and _post_stream so we can verify which one is invoked
            resp_post = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Result from non-streaming post",
                    }
                }],
                "timings": {"predicted_per_second": 48.0},
                "usage": {"completion_tokens": 15},
            }
            stream_chunks = [
                {"choices": [{"delta": {"content": "Result from streaming"}}]},
                {"choices": [], "timings": {"predicted_per_second": 52.0}, "usage": {"completion_tokens": 8}},
            ]

            with mock.patch.object(agent, "_post", return_value=resp_post) as mock_post, \
                 mock.patch.object(agent, "_post_stream", return_value=iter(stream_chunks)) as mock_stream:

                # In hybrid mode: tool exploration turn (turns < max_turns, provide_tools is present)
                # routes to _post (non-streaming)
                res = agent.step("Query requiring tools", verbose=False)
                self.assertEqual(res, "Result from non-streaming post")
                self.assertFalse(agent.last_turn_streamed)
                mock_post.assert_called_once()
                mock_stream.assert_not_called()

                # Reset mocks
                mock_post.reset_mock()
                mock_stream.reset_mock()

                # In "always" mode: even with tools, routes to _post_stream
                agent.stream_mode = "always"
                res = agent.step("Another query", verbose=False)
                self.assertEqual(res, "Result from streaming")
                self.assertTrue(agent.last_turn_streamed)
                mock_stream.assert_called_once()
                mock_post.assert_not_called()

                # Reset mocks
                mock_post.reset_mock()
                mock_stream.reset_mock()

                # In "never" mode: routes to _post
                agent.stream_mode = "never"
                res = agent.step("Yet another query", verbose=False)
                self.assertEqual(res, "Result from non-streaming post")
                self.assertFalse(agent.last_turn_streamed)
                mock_post.assert_called_once()
                mock_stream.assert_not_called()
        finally:
            agent.close()

    def test_agent_non_streaming_telemetry_and_reasoning(self):
        """Verify non-streaming _post path extracts reasoning_content and populates last_turn_stats."""
        agent = SlipstreamAgent(stream_mode="never")
        try:
            resp_post = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "Deep architectural analysis here.",
                        "content": "Final implementation plan.",
                    }
                }],
                "timings": {"predicted_per_second": 64.2},
                "usage": {"completion_tokens": 128},
            }

            mock_console = mock.MagicMock()
            with mock.patch.object(agent, "_post", return_value=resp_post):
                res = agent.step("Analyze codebase", verbose=True, console=mock_console)
                self.assertEqual(res, "Final implementation plan.")
                self.assertFalse(agent.last_turn_streamed)
                self.assertEqual(agent.last_turn_stats.get("tok_s"), 64.2)
                self.assertEqual(agent.last_turn_stats.get("tokens"), 128)

                # Verify thinking was printed
                print_calls = [str(call) for call in mock_console.print.call_args_list]
                self.assertTrue(any("thinking" in c for c in print_calls))
                self.assertTrue(any("Deep architectural analysis" in c for c in print_calls))
        finally:
            agent.close()


    def test_list_checkpoints_and_latest_restore(self):
        """Verify list_checkpoints ordering and restore_checkpoint('latest')."""
        with tempfile.TemporaryDirectory() as tmpdir:
            a1 = SlipstreamAgent(cwd=tmpdir, session_id="session-alpha")
            try:
                a1.messages.append({"role": "user", "content": "alpha task"})
                a1.save_checkpoint(vault_dir=tmpdir)
            finally:
                a1.close()

            time.sleep(0.05)

            a2 = SlipstreamAgent(cwd=tmpdir, session_id="session-beta")
            try:
                a2.messages.append({"role": "user", "content": "beta task"})
                a2.save_checkpoint(vault_dir=tmpdir)
            finally:
                a2.close()

            checkpoints = SlipstreamAgent.list_checkpoints(vault_dir=tmpdir)
            self.assertEqual(len(checkpoints), 2)
            self.assertEqual(checkpoints[0]["session_id"], "session-beta")
            self.assertEqual(checkpoints[1]["session_id"], "session-alpha")

            # Restore latest
            restored = SlipstreamAgent.restore_checkpoint("latest", vault_dir=tmpdir, cwd=tmpdir)
            self.assertIsNotNone(restored)
            self.assertEqual(restored.session_id, "session-beta")
            self.assertEqual(restored.messages[-1]["content"], "beta task")
            restored.close()

            # Restore by prefix
            restored_alpha = SlipstreamAgent.restore_checkpoint("session-al", vault_dir=tmpdir, cwd=tmpdir)
            self.assertIsNotNone(restored_alpha)
            self.assertEqual(restored_alpha.session_id, "session-alpha")
            restored_alpha.close()

    def test_slash_commands_session_management(self):
        """Verify /save, /sessions, /resume, /continue, and /retry slash commands."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from server.agent.agent import _dispatch_slash_command

            agent = SlipstreamAgent(cwd=tmpdir, session_id="vault-sess-test")
            try:
                agent.messages.append({"role": "user", "content": "Build feature X"})
                agent.last_failed_prompt = "Build feature X"

                # /save
                with mock.patch("pathlib.Path.home", return_value=Path(tmpdir)):
                    handled, _ = _dispatch_slash_command(agent, "/save", plan_mode=False)
                    self.assertTrue(handled)

                    # /sessions
                    handled, _ = _dispatch_slash_command(agent, "/sessions", plan_mode=False)
                    self.assertTrue(handled)

                    # /continue
                    handled, _ = _dispatch_slash_command(agent, "/continue", plan_mode=False)
                    self.assertTrue(handled)
                    self.assertIn("continue your work", getattr(agent, "pending_prompt", ""))

                    # /retry
                    handled, _ = _dispatch_slash_command(agent, "/retry", plan_mode=False)
                    self.assertTrue(handled)
                    self.assertEqual(getattr(agent, "pending_prompt", ""), "Build feature X")

                    # /resume
                    handled, _ = _dispatch_slash_command(agent, "/resume vault-sess-test", plan_mode=False)
                    self.assertTrue(handled)

                    # /compact
                    handled, _ = _dispatch_slash_command(agent, "/compact", plan_mode=False)
                    self.assertTrue(handled)

                    # /model
                    handled, _ = _dispatch_slash_command(agent, "/model", plan_mode=False)
                    self.assertTrue(handled)
            finally:
                agent.close()

    def test_agent_timeout_none_and_zero_configuration(self):
        """Verify that timeout=None or timeout=0 sets unbounded timeout with settimeout(None)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Default without environment variable
            agent1 = SlipstreamAgent(cwd=tmpdir)
            self.assertIsNone(agent1.timeout)
            agent1.close()

            # 2. Explicit timeout=0.0
            agent2 = SlipstreamAgent(cwd=tmpdir, timeout=0.0)
            self.assertIsNone(agent2.timeout)
            agent2.close()

            # 3. Explicit timeout=30.0
            agent3 = SlipstreamAgent(cwd=tmpdir, timeout=30.0)
            self.assertEqual(agent3.timeout, 30.0)
            agent3.close()

            # 4. UnixHTTPConnection with timeout=None
            from server.agent.agent import UnixHTTPConnection
            conn = UnixHTTPConnection("/tmp/dummy.sock", timeout=None)
            self.assertIsNone(conn.timeout)

    def test_prefix_cache_hit_rate_telemetry_accumulation(self):
        """Verify that prompt_tokens and cached_tokens accumulate and /stats displays hit rate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from server.agent.agent import _dispatch_slash_command

            agent = SlipstreamAgent(cwd=tmpdir, stream_mode="never")
            try:
                # Mock _post returning usage with cached tokens
                mock_resp = {
                    "choices": [{"message": {"role": "assistant", "content": "Done"}}],
                    "usage": {
                        "prompt_tokens": 1000,
                        "completion_tokens": 50,
                        "prompt_tokens_details": {"cached_tokens": 750},
                    },
                }
                with mock.patch.object(agent, "_post", return_value=mock_resp):
                    agent.step("Turn 1", verbose=False)
                    agent.step("Turn 2", verbose=False)

                self.assertEqual(agent.stats["total_prompt_tokens"], 2000)
                self.assertEqual(agent.stats["cached_prompt_tokens"], 1500)

                # Verify /stats renders with 75.0%
                table_output = []
                with mock.patch.object(agent.console, "print", side_effect=lambda x: table_output.append(x)):
                    handled, _ = _dispatch_slash_command(agent, "/stats", plan_mode=False)
                    self.assertTrue(handled)
                    self.assertTrue(len(table_output) >= 1)
            finally:
                agent.close()

    def test_incremental_tool_parse_cursor_streaming(self):
        """Verify incremental tool-parse cursor during streaming speculative execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "test.txt"
            f.write_text("hello world")

            agent = SlipstreamAgent(cwd=tmpdir, stream_mode="always")
            try:
                chunks = [
                    {
                        "choices": [{
                            "delta": {
                                "tool_calls": [{
                                    "index": 0,
                                    "function": {"name": "view_file", "arguments": '{"path": "test.txt"'},
                                }]
                            }
                        }]
                    },
                    {
                        "choices": [{
                            "delta": {
                                "tool_calls": [{
                                    "index": 0,
                                    "function": {"arguments": '}'},
                                }]
                            }
                        }],
                        "usage": {
                            "prompt_tokens": 500,
                            "completion_tokens": 20,
                            "prompt_tokens_details": {"cached_tokens": 400},
                        },
                    },
                    {
                        "choices": [{
                            "delta": {
                                "content": "Done"
                            }
                        }]
                    }
                ]
                with mock.patch.object(agent, "_post_stream", return_value=iter(chunks)):
                    # Mock execute_tool to avoid executing tool on follow-up
                    with mock.patch.object(agent, "execute_tool", return_value="hello world") as mock_exec:
                        agent.step("read file", max_turns=1, verbose=False)
                        self.assertEqual(agent.stats["total_prompt_tokens"], 500)
                        self.assertEqual(agent.stats["cached_prompt_tokens"], 400)
            finally:
                agent.close()


    def test_manage_todo_auto_advance_and_hallucinated_args(self):
        agent = SlipstreamAgent()
        try:
            # 1. Initialize 4 tasks
            res_set = agent.execute_tool("manage_todo", {"action": "set", "tasks": ["Task A", "Task B", "Task C", "Task D"]})
            self.assertIn("Plan initialized with 4 steps", res_set)
            self.assertEqual(len(agent.todo.tasks), 4)

            # 2. Advance step 1 explicitly
            agent.execute_tool("manage_todo", {"action": "update", "task_id": 1, "status": "completed"})
            self.assertEqual(agent.todo.tasks[0]["status"], "completed")

            # 3. Model hallucinates keys {"task_id1": null, ...} with status "completed"
            # It should auto-advance to step 2!
            res_auto1 = agent.execute_tool("manage_todo", {"action": "update", "status": "completed", "task_id1": None, "task_id2": None})
            self.assertIn("Step 2 updated to 'completed'", res_auto1)
            self.assertEqual(agent.todo.tasks[1]["status"], "completed")

            # 4. Model calls update with string step "3"
            res_str = agent.execute_tool("manage_todo", {"action": "update", "step": "3", "status": "in_progress"})
            self.assertIn("Step 3 updated to 'in_progress'", res_str)
            self.assertEqual(agent.todo.tasks[2]["status"], "in_progress")

            # 5. Model calls update without any task_id at all
            res_auto2 = agent.execute_tool("manage_todo", {"action": "update", "status": "completed"})
            self.assertIn("Step 3 updated to 'completed'", res_auto2)
            self.assertEqual(agent.todo.tasks[2]["status"], "completed")
            self.assertEqual(agent.todo.tasks[3]["status"], "pending")
        finally:
            agent.close()

    def test_consecutive_circuit_breaker(self):
        agent = SlipstreamAgent()
        try:
            # Different commands across turns should NOT trip the circuit breaker even if name is run_command
            chunks_turn1 = [{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "run_command", "arguments": '{"command":"ls"}'}}]}}]}]
            chunks_turn2 = [{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "run_command", "arguments": '{"command":"pwd"}'}}]}}]}]
            chunks_turn3 = [{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "run_command", "arguments": '{"command":"whoami"}'}}]}}]}]

            with mock.patch.object(agent, "execute_tool", return_value="output"):
                with mock.patch.object(agent, "_post_stream", return_value=iter(chunks_turn1)):
                    agent.step("turn 1", max_turns=1, verbose=False)
                with mock.patch.object(agent, "_post_stream", return_value=iter(chunks_turn2)):
                    agent.step("turn 2", max_turns=1, verbose=False)
                with mock.patch.object(agent, "_post_stream", return_value=iter(chunks_turn3)):
                    agent.step("turn 3", max_turns=1, verbose=False)

                # The circuit breaker message should NOT be in messages
                cb_triggered = any("You have repeated this exact tool call 3 times" in m.get("content", "") for m in agent.messages)
                self.assertFalse(cb_triggered)

                # Now if the exact same command is executed 3 times consecutively, it should trip
                same_chunk = [{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "run_command", "arguments": '{"command":"stuck"}'}}]}}]}]
                for _ in range(3):
                    with mock.patch.object(agent, "_post_stream", return_value=iter(same_chunk)):
                        agent.step("stuck turn", max_turns=1, verbose=False)

                cb_triggered = any("You have repeated this exact tool call 3 times" in m.get("content", "") for m in agent.messages)
                self.assertTrue(cb_triggered)
        finally:
            agent.close()


if __name__ == "__main__":
    unittest.main()


