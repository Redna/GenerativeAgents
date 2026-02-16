import unittest
from unittest.mock import MagicMock, patch
import dspy
from generative_agents.conversational.pipelines.poignance import rate_poignance
from generative_agents.conversational.pipelines.decide_to_talk import decide_to_talk

class TestDSPyPipelines(unittest.TestCase):
    def setUp(self):
        # We need to mock dspy.ChainOfThought because dspy 3.x removed DummyLM or changed API
        self.patcher = patch('dspy.ChainOfThought')
        self.mock_cot = self.patcher.start()
        
    def tearDown(self):
        self.patcher.stop()

    def test_rate_poignance(self):
        # Setup mock return value
        mock_instance = MagicMock()
        mock_instance.return_value.rating = 8
        self.mock_cot.return_value = mock_instance

        # Run function
        rating = rate_poignance("Agent", "Identity", "Event", "Description")
        
        # Verify
        self.assertEqual(rating, 8)
        self.mock_cot.assert_called()

    def test_decide_to_talk(self):
        # Setup mock return value
        mock_instance = MagicMock()
        mock_instance.return_value.initiate_conversation = True
        self.mock_cot.return_value = mock_instance

        # Run function
        result = decide_to_talk("Context", "Time", "Me", "You", "Summary", "ObsMe", "ObsYou")
        
        # Verify
        self.assertTrue(result)
        self.mock_cot.assert_called()

    def test_hourly_breakdown(self):
        # Mock for `predict(name=..., ...)` returning an object with `schedule` attribute
        mock_response = MagicMock()
        mock_response.schedule = [
            MagicMock(time="08:00 AM", activity="Wake up"),
            MagicMock(time="09:00 AM", activity="Work")
        ]
        
        # We assume ChainOfThought is used.
        # Check if the code uses dspy.ChainOfThought(HourlyScheduleSignature)
        # Since ChainOfThought is already mocked in setUp as self.mock_cot (class mock)
        # We need to ensure that when called with HourlyScheduleSignature, it returns a mock that returns mock_response
        
        # In setUp: self.mock_cot = patch('dspy.ChainOfThought').start()
        # In code: predict = dspy.ChainOfThought(HourlyScheduleSignature) -> returns instance
        #          response = predict(...) -> returns result
        
        # So we configure the return value of the instance
        self.mock_cot.return_value.return_value = mock_response
        
        from generative_agents.conversational.pipelines.hourly_breakdown import create_hourly_schedule
        
        result = create_hourly_schedule("Agent", "Identity", [{"time": "08:00 AM", "activity": "Wake up"}], "08:00 AM")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 24)
        self.assertEqual(result[8]['activity'], "Wake up")

    def test_task_decomposition(self):
        mock_response = MagicMock()
        mock_response.subtasks = [
            MagicMock(activity_name="Task A", duration_minutes=30),
            MagicMock(activity_name="Task B", duration_minutes=30)
        ]
        
        # Same as above, configure mock_cot instance return value
        self.mock_cot.return_value.return_value = mock_response
        
        from generative_agents.conversational.pipelines.task_decomposition import create_decomposition_schedule
        result = create_decomposition_schedule("Agent", "Id", "Task", "9:00", "10:00", 60, "Today", "Ctx")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], "Task A")

    def test_action_location_arena(self):
        mock_response = MagicMock()
        mock_response.next_area = "kitchen"
        # Class() -> instance. instance() -> result.
        self.mock_cot.return_value.return_value = mock_response
        
        from generative_agents.conversational.pipelines.action_location_arena import action_area_locations
        result = action_area_locations("Agent", "bedroom", "Sector", "Other", "kitchen, bathroom", "Cooking")
        self.assertEqual(result, "kitchen")

    def test_action_location_sector(self):
        mock_response = MagicMock()
        mock_response.next_sector = "Park"
        self.mock_cot.return_value.return_value = mock_response
        
        from generative_agents.conversational.pipelines.action_location_sector import action_sector_locations
        result = action_sector_locations("Agent", "Home", "Rooms", "Current", "Rooms", "Park, Shop", "Walk")
        self.assertEqual(result, "Park")
        
    def test_action_location_object(self):
        mock_response = MagicMock()
        mock_response.next_object = "bed"
        
        with patch('dspy.Predict') as mock_predict:
            mock_predict.return_value.return_value = mock_response
            from generative_agents.conversational.pipelines.action_location_game_object import action_location_game_object
            
            result = action_location_game_object("Sleep", "bed, table")
            self.assertEqual(result, "bed")

    def test_action_event_triple(self):
        mock_response = MagicMock()
        mock_response.subject = "John"
        mock_response.predicate = "is"
        mock_response.object = "sleeping"
        self.mock_cot.return_value.return_value = mock_response
        
        from generative_agents.conversational.pipelines.action_event_tripple import action_event_triple
        subject, predicate, obj = action_event_triple("John", "Sleeping", "Other")
        self.assertEqual(subject, "Other") # Address override
        self.assertEqual(predicate, "is")
        self.assertEqual(obj, "sleeping")

    def test_action_pronunciatio(self):
        mock_response = MagicMock()
        mock_response.emoji = "😴"
        self.mock_cot.return_value.return_value = mock_response
        
        from generative_agents.conversational.pipelines.action_pronunciatio import action_pronunciatio
        result = action_pronunciatio("Sleeping")
        self.assertEqual(result, "😴")

    def test_contextualize_event(self):
        mock_response = MagicMock()
        mock_response.event_context = "Context"
        self.mock_cot.return_value.return_value = mock_response
        
        from generative_agents.conversational.pipelines.contextualize_event import contextualize_event
        result = contextualize_event("Agent", "Id", "Event", "Events", "Thoughts")
        self.assertEqual(result, "Context")

    def test_conversation(self):
        mock_response = MagicMock()
        mock_response.utterance = "Hi"
        mock_response.end_conversation = False
        self.mock_cot.return_value.return_value = mock_response
        
        from generative_agents.conversational.pipelines.conversation import run_conversation
        utt, end = run_conversation("Agent", "Id", "Mem", "Ctx", "Loc", "Act", "With", "Act2", "Hist")
        self.assertEqual(utt, "Hi")
        self.assertFalse(end)
        
    def test_conversation_summary(self):
        mock_response = MagicMock()
        mock_response.summary = "Summary"
        self.mock_cot.return_value.return_value = mock_response
        
        from generative_agents.conversational.pipelines.conversation_summary import conversation_summary
        result = conversation_summary("Dialog")
        self.assertEqual(result, "Summary")


    def test_evidence_and_insights(self):
        mock_response = MagicMock()
        mock_response.insights = ["Insight 1", "Insight 2"]
        self.mock_cot.return_value.return_value = mock_response
        
        from generative_agents.conversational.pipelines.evidence_and_insights import evidence_and_insights
        result = evidence_and_insights(["Statement 1"], 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "Insight 1")

    def test_first_daily_plan(self):
        mock_response = MagicMock()
        # Mocking list of PlanOutline objects
        item1 = MagicMock()
        item1.hour = 8
        item1.description = "Wake up"
        mock_response.plan_in_broad_strokes = [item1]
        self.mock_cot.return_value.return_value = mock_response
        
        from generative_agents.conversational.pipelines.first_daily_plan import create_daily_plan
        result = create_daily_plan("Name", "Id", "Today", "8:00 AM")
        self.assertIn("08:00 AM", result)
        self.assertEqual(result["08:00 AM"], "Wake up")

    def test_identity(self):
        mock_response = MagicMock()
        mock_response.identity = "Identity desc"
        self.mock_cot.return_value.return_value = mock_response
        
        from generative_agents.conversational.pipelines.identity import formulate_identity
        result = formulate_identity("Name", "Context")
        self.assertEqual(result, "Identity desc")

    def test_memo_on_conversation(self):
        mock_response = MagicMock()
        mock_response.memo = "Interesting point"
        self.mock_cot.return_value.return_value = mock_response
        
        from generative_agents.conversational.pipelines.memo_on_conversation import memo_on_conversation
        result = memo_on_conversation("Name", "Conversation")
        self.assertEqual(result, "Interesting point")

    def test_new_decomposition_schedule(self):
        mock_response = MagicMock()
        task1 = MagicMock()
        task1.time = "09:00 AM"
        task1.activity = "Work"
        mock_response.schedule = [task1]
        self.mock_cot.return_value.return_value = mock_response
        
        from generative_agents.conversational.pipelines.new_decomposition_schedule import create_new_decomposition_schedule
        # Helper to create dummy schedule slice
        slice_input = [("Old Task", 60)]
        result = create_new_decomposition_schedule("Name", 9, 10, "New Event", 30, slice_input)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "Work")
        # duration is placeholder 60
        self.assertEqual(result[0][1], 60)

    def test_object_event(self):
        mock_response = MagicMock()
        mock_response.state = "clean"
        self.mock_cot.return_value.return_value = mock_response
        
        from generative_agents.conversational.pipelines.object_event import describe_object_state
        desc, tuple_res = describe_object_state("Name", "Obj", "Addr", "Action")
        self.assertIn("clean", desc)
        self.assertEqual(tuple_res[2], "clean")

    def test_planning_on_conversation(self):
        mock_response = MagicMock()
        mock_response.to_remember = "Remember this."
        self.mock_cot.return_value.return_value = mock_response
        
        from generative_agents.conversational.pipelines.planning_on_conversation import planning_on_conversation
        result = planning_on_conversation("Agent", "Chat")
        self.assertEqual(result, "Remember this.")

    def test_reflection_points(self):
        mock_response = MagicMock()
        mock_response.questions = ["Q1?", "Q2?"]
        self.mock_cot.return_value.return_value = mock_response
        
        from generative_agents.conversational.pipelines.reflection_points import reflection_points
        result = reflection_points("Memory", 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "Q1?")

    def test_summarize_chat_relationship(self):
        mock_response = MagicMock()
        mock_response.relationship_summary = "Friends"
        self.mock_cot.return_value.return_value = mock_response
        
        from generative_agents.conversational.pipelines.summarize_chat_relationship import summarize_chat_relationship
        result = summarize_chat_relationship("Stmts", "A1", "A2")
        self.assertEqual(result, "Friends")

if __name__ == '__main__':
    unittest.main()
