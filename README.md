
```mermaid
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
    __start__([<p>__start__</p>]):::first
    round_update(round_update)
    reflect_changes(reflect_changes)
    Abigail_Chen_set_daytype_-_Abigail_Chen(set_daytype - Abigail Chen)
    Abigail_Chen_perception_-_Abigail_Chen_perceive_space(perceive_space)
    Abigail_Chen_perception_-_Abigail_Chen_perceive_events(perceive_events)
    Abigail_Chen_perception_-_Abigail_Chen_store_events(store_events)
    Abigail_Chen_retrieval_-_Abigail_Chen_add_current_event(add_current_event)
    Abigail_Chen_retrieval_-_Abigail_Chen_retrieve_events(retrieve_events)
    Abigail_Chen_retrieval_-_Abigail_Chen_retrieve_thoughts(retrieve_thoughts)
    Abigail_Chen_retrieval_-_Abigail_Chen___end__(<p>__end__</p>)
    Abigail_Chen_plan_-_Abigail_Chen_long_term_planning(long_term_planning)
    Abigail_Chen_plan_-_Abigail_Chen_determine_action(determine_action)
    Abigail_Chen_plan_-_Abigail_Chen_choose_retrieved(choose_retrieved)
    Abigail_Chen_plan_-_Abigail_Chen_react(react)
    Abigail_Chen_plan_-_Abigail_Chen_wrap_up(wrap_up)
    Abigail_Chen_execution_-_Abigail_Chen(execution - Abigail Chen)
    Abigail_Chen_reflection_-_Abigail_Chen___start__(<p>__start__</p>)
    Abigail_Chen_reflection_-_Abigail_Chen_reflect(reflect)
    Abigail_Chen_reflection_-_Abigail_Chen_retrieve_last_conversation(retrieve_last_conversation)
    Abigail_Chen_reflection_-_Abigail_Chen_reflect_on_conversation(reflect_on_conversation)
    Abigail_Chen_reflection_-_Abigail_Chen___end__(<p>__end__</p>)
    Abigail_Chen_wrap_up_-_Abigail_Chen(wrap_up - Abigail Chen)
    Yuriko_Yamamoto_set_daytype_-_Yuriko_Yamamoto(set_daytype - Yuriko Yamamoto)
    Yuriko_Yamamoto_perception_-_Yuriko_Yamamoto_perceive_space(perceive_space)
    Yuriko_Yamamoto_perception_-_Yuriko_Yamamoto_perceive_events(perceive_events)
    Yuriko_Yamamoto_perception_-_Yuriko_Yamamoto_store_events(store_events)
    Yuriko_Yamamoto_retrieval_-_Yuriko_Yamamoto_add_current_event(add_current_event)
    Yuriko_Yamamoto_retrieval_-_Yuriko_Yamamoto_retrieve_events(retrieve_events)
    Yuriko_Yamamoto_retrieval_-_Yuriko_Yamamoto_retrieve_thoughts(retrieve_thoughts)
    Yuriko_Yamamoto_retrieval_-_Yuriko_Yamamoto___end__(<p>__end__</p>)
    Yuriko_Yamamoto_plan_-_Yuriko_Yamamoto_long_term_planning(long_term_planning)
    Yuriko_Yamamoto_plan_-_Yuriko_Yamamoto_determine_action(determine_action)
    Yuriko_Yamamoto_plan_-_Yuriko_Yamamoto_choose_retrieved(choose_retrieved)
    Yuriko_Yamamoto_plan_-_Yuriko_Yamamoto_react(react)
    Yuriko_Yamamoto_plan_-_Yuriko_Yamamoto_wrap_up(wrap_up)
    Yuriko_Yamamoto_execution_-_Yuriko_Yamamoto(execution - Yuriko Yamamoto)
    Yuriko_Yamamoto_reflection_-_Yuriko_Yamamoto___start__(<p>__start__</p>)
    Yuriko_Yamamoto_reflection_-_Yuriko_Yamamoto_reflect(reflect)
    Yuriko_Yamamoto_reflection_-_Yuriko_Yamamoto_retrieve_last_conversation(retrieve_last_conversation)
    Yuriko_Yamamoto_reflection_-_Yuriko_Yamamoto_reflect_on_conversation(reflect_on_conversation)
    Yuriko_Yamamoto_reflection_-_Yuriko_Yamamoto___end__(<p>__end__</p>)
    Yuriko_Yamamoto_wrap_up_-_Yuriko_Yamamoto(wrap_up - Yuriko Yamamoto)
    Tamara_Taylor_set_daytype_-_Tamara_Taylor(set_daytype - Tamara Taylor)
    Tamara_Taylor_perception_-_Tamara_Taylor_perceive_space(perceive_space)
    Tamara_Taylor_perception_-_Tamara_Taylor_perceive_events(perceive_events)
    Tamara_Taylor_perception_-_Tamara_Taylor_store_events(store_events)
    Tamara_Taylor_retrieval_-_Tamara_Taylor_add_current_event(add_current_event)
    Tamara_Taylor_retrieval_-_Tamara_Taylor_retrieve_events(retrieve_events)
    Tamara_Taylor_retrieval_-_Tamara_Taylor_retrieve_thoughts(retrieve_thoughts)
    Tamara_Taylor_retrieval_-_Tamara_Taylor___end__(<p>__end__</p>)
    Tamara_Taylor_plan_-_Tamara_Taylor_long_term_planning(long_term_planning)
    Tamara_Taylor_plan_-_Tamara_Taylor_determine_action(determine_action)
    Tamara_Taylor_plan_-_Tamara_Taylor_choose_retrieved(choose_retrieved)
    Tamara_Taylor_plan_-_Tamara_Taylor_react(react)
    Tamara_Taylor_plan_-_Tamara_Taylor_wrap_up(wrap_up)
    Tamara_Taylor_execution_-_Tamara_Taylor(execution - Tamara Taylor)
    Tamara_Taylor_reflection_-_Tamara_Taylor___start__(<p>__start__</p>)
    Tamara_Taylor_reflection_-_Tamara_Taylor_reflect(reflect)
    Tamara_Taylor_reflection_-_Tamara_Taylor_retrieve_last_conversation(retrieve_last_conversation)
    Tamara_Taylor_reflection_-_Tamara_Taylor_reflect_on_conversation(reflect_on_conversation)
    Tamara_Taylor_reflection_-_Tamara_Taylor___end__(<p>__end__</p>)
    Tamara_Taylor_wrap_up_-_Tamara_Taylor(wrap_up - Tamara Taylor)
    __end__([<p>__end__</p>]):::last
    Abigail_Chen_wrap_up_-_Abigail_Chen --> reflect_changes;
    Tamara_Taylor_wrap_up_-_Tamara_Taylor --> reflect_changes;
    Yuriko_Yamamoto_wrap_up_-_Yuriko_Yamamoto --> reflect_changes;
    __start__ --> round_update;
    round_update -.-> Abigail_Chen_set_daytype_-_Abigail_Chen;
    round_update -.-> Yuriko_Yamamoto_set_daytype_-_Yuriko_Yamamoto;
    round_update -.-> Tamara_Taylor_set_daytype_-_Tamara_Taylor;
    reflect_changes -. &nbsp;True&nbsp; .-> __end__;
    reflect_changes -. &nbsp;False&nbsp; .-> round_update;
    subgraph Abigail Chen
    Abigail_Chen_execution_-_Abigail_Chen --> Abigail_Chen_reflection_-_Abigail_Chen___start__;
    Abigail_Chen_perception_-_Abigail_Chen_store_events --> Abigail_Chen_retrieval_-_Abigail_Chen_add_current_event;
    Abigail_Chen_plan_-_Abigail_Chen_wrap_up --> Abigail_Chen_execution_-_Abigail_Chen;
    Abigail_Chen_reflection_-_Abigail_Chen___end__ --> Abigail_Chen_wrap_up_-_Abigail_Chen;
    Abigail_Chen_retrieval_-_Abigail_Chen___end__ --> Abigail_Chen_plan_-_Abigail_Chen_long_term_planning;
    Abigail_Chen_set_daytype_-_Abigail_Chen --> Abigail_Chen_perception_-_Abigail_Chen_perceive_space;
    subgraph perception - Abigail Chen
    Abigail_Chen_perception_-_Abigail_Chen_perceive_space --> Abigail_Chen_perception_-_Abigail_Chen_perceive_events;
    Abigail_Chen_perception_-_Abigail_Chen_perceive_events -.-> Abigail_Chen_perception_-_Abigail_Chen_store_events;
    end
    subgraph retrieval - Abigail Chen
    Abigail_Chen_retrieval_-_Abigail_Chen_add_current_event --> Abigail_Chen_retrieval_-_Abigail_Chen_retrieve_events;
    Abigail_Chen_retrieval_-_Abigail_Chen_add_current_event --> Abigail_Chen_retrieval_-_Abigail_Chen_retrieve_thoughts;
    Abigail_Chen_retrieval_-_Abigail_Chen_retrieve_events --> Abigail_Chen_retrieval_-_Abigail_Chen___end__;
    Abigail_Chen_retrieval_-_Abigail_Chen_retrieve_thoughts --> Abigail_Chen_retrieval_-_Abigail_Chen___end__;
    end
    subgraph plan - Abigail Chen
    Abigail_Chen_plan_-_Abigail_Chen_choose_retrieved --> Abigail_Chen_plan_-_Abigail_Chen_react;
    Abigail_Chen_plan_-_Abigail_Chen_determine_action --> Abigail_Chen_plan_-_Abigail_Chen_choose_retrieved;
    Abigail_Chen_plan_-_Abigail_Chen_long_term_planning --> Abigail_Chen_plan_-_Abigail_Chen_determine_action;
    Abigail_Chen_plan_-_Abigail_Chen_react --> Abigail_Chen_plan_-_Abigail_Chen_wrap_up;
    end
    subgraph reflection - Abigail Chen
    Abigail_Chen_reflection_-_Abigail_Chen___start__ --> Abigail_Chen_reflection_-_Abigail_Chen_retrieve_last_conversation;
    Abigail_Chen_reflection_-_Abigail_Chen_reflect --> Abigail_Chen_reflection_-_Abigail_Chen___end__;
    Abigail_Chen_reflection_-_Abigail_Chen_reflect_on_conversation --> Abigail_Chen_reflection_-_Abigail_Chen___end__;
    Abigail_Chen_reflection_-_Abigail_Chen___start__ -. &nbsp;True&nbsp; .-> Abigail_Chen_reflection_-_Abigail_Chen_reflect;
    Abigail_Chen_reflection_-_Abigail_Chen___start__ -. &nbsp;False&nbsp; .-> Abigail_Chen_reflection_-_Abigail_Chen___end__;
    Abigail_Chen_reflection_-_Abigail_Chen_retrieve_last_conversation -. &nbsp;True&nbsp; .-> Abigail_Chen_reflection_-_Abigail_Chen_reflect_on_conversation;
    Abigail_Chen_reflection_-_Abigail_Chen_retrieve_last_conversation -. &nbsp;False&nbsp; .-> Abigail_Chen_reflection_-_Abigail_Chen___end__;
    end
    end
    subgraph Yuriko Yamamoto
    Yuriko_Yamamoto_execution_-_Yuriko_Yamamoto --> Yuriko_Yamamoto_reflection_-_Yuriko_Yamamoto___start__;
    Yuriko_Yamamoto_perception_-_Yuriko_Yamamoto_store_events --> Yuriko_Yamamoto_retrieval_-_Yuriko_Yamamoto_add_current_event;
    Yuriko_Yamamoto_plan_-_Yuriko_Yamamoto_wrap_up --> Yuriko_Yamamoto_execution_-_Yuriko_Yamamoto;
    Yuriko_Yamamoto_reflection_-_Yuriko_Yamamoto___end__ --> Yuriko_Yamamoto_wrap_up_-_Yuriko_Yamamoto;
    Yuriko_Yamamoto_retrieval_-_Yuriko_Yamamoto___end__ --> Yuriko_Yamamoto_plan_-_Yuriko_Yamamoto_long_term_planning;
    Yuriko_Yamamoto_set_daytype_-_Yuriko_Yamamoto --> Yuriko_Yamamoto_perception_-_Yuriko_Yamamoto_perceive_space;
    subgraph perception - Yuriko Yamamoto
    Yuriko_Yamamoto_perception_-_Yuriko_Yamamoto_perceive_space --> Yuriko_Yamamoto_perception_-_Yuriko_Yamamoto_perceive_events;
    Yuriko_Yamamoto_perception_-_Yuriko_Yamamoto_perceive_events -.-> Yuriko_Yamamoto_perception_-_Yuriko_Yamamoto_store_events;
    end
    subgraph retrieval - Yuriko Yamamoto
    Yuriko_Yamamoto_retrieval_-_Yuriko_Yamamoto_add_current_event --> Yuriko_Yamamoto_retrieval_-_Yuriko_Yamamoto_retrieve_events;
    Yuriko_Yamamoto_retrieval_-_Yuriko_Yamamoto_add_current_event --> Yuriko_Yamamoto_retrieval_-_Yuriko_Yamamoto_retrieve_thoughts;
    Yuriko_Yamamoto_retrieval_-_Yuriko_Yamamoto_retrieve_events --> Yuriko_Yamamoto_retrieval_-_Yuriko_Yamamoto___end__;
    Yuriko_Yamamoto_retrieval_-_Yuriko_Yamamoto_retrieve_thoughts --> Yuriko_Yamamoto_retrieval_-_Yuriko_Yamamoto___end__;
    end
    subgraph plan - Yuriko Yamamoto
    Yuriko_Yamamoto_plan_-_Yuriko_Yamamoto_choose_retrieved --> Yuriko_Yamamoto_plan_-_Yuriko_Yamamoto_react;
    Yuriko_Yamamoto_plan_-_Yuriko_Yamamoto_determine_action --> Yuriko_Yamamoto_plan_-_Yuriko_Yamamoto_choose_retrieved;
    Yuriko_Yamamoto_plan_-_Yuriko_Yamamoto_long_term_planning --> Yuriko_Yamamoto_plan_-_Yuriko_Yamamoto_determine_action;
    Yuriko_Yamamoto_plan_-_Yuriko_Yamamoto_react --> Yuriko_Yamamoto_plan_-_Yuriko_Yamamoto_wrap_up;
    end
    subgraph reflection - Yuriko Yamamoto
    Yuriko_Yamamoto_reflection_-_Yuriko_Yamamoto___start__ --> Yuriko_Yamamoto_reflection_-_Yuriko_Yamamoto_retrieve_last_conversation;
    Yuriko_Yamamoto_reflection_-_Yuriko_Yamamoto_reflect --> Yuriko_Yamamoto_reflection_-_Yuriko_Yamamoto___end__;
    Yuriko_Yamamoto_reflection_-_Yuriko_Yamamoto_reflect_on_conversation --> Yuriko_Yamamoto_reflection_-_Yuriko_Yamamoto___end__;
    Yuriko_Yamamoto_reflection_-_Yuriko_Yamamoto___start__ -. &nbsp;True&nbsp; .-> Yuriko_Yamamoto_reflection_-_Yuriko_Yamamoto_reflect;
    Yuriko_Yamamoto_reflection_-_Yuriko_Yamamoto___start__ -. &nbsp;False&nbsp; .-> Yuriko_Yamamoto_reflection_-_Yuriko_Yamamoto___end__;
    Yuriko_Yamamoto_reflection_-_Yuriko_Yamamoto_retrieve_last_conversation -. &nbsp;True&nbsp; .-> Yuriko_Yamamoto_reflection_-_Yuriko_Yamamoto_reflect_on_conversation;
    Yuriko_Yamamoto_reflection_-_Yuriko_Yamamoto_retrieve_last_conversation -. &nbsp;False&nbsp; .-> Yuriko_Yamamoto_reflection_-_Yuriko_Yamamoto___end__;
    end
    end
    subgraph Tamara Taylor
    Tamara_Taylor_execution_-_Tamara_Taylor --> Tamara_Taylor_reflection_-_Tamara_Taylor___start__;
    Tamara_Taylor_perception_-_Tamara_Taylor_store_events --> Tamara_Taylor_retrieval_-_Tamara_Taylor_add_current_event;
    Tamara_Taylor_plan_-_Tamara_Taylor_wrap_up --> Tamara_Taylor_execution_-_Tamara_Taylor;
    Tamara_Taylor_reflection_-_Tamara_Taylor___end__ --> Tamara_Taylor_wrap_up_-_Tamara_Taylor;
    Tamara_Taylor_retrieval_-_Tamara_Taylor___end__ --> Tamara_Taylor_plan_-_Tamara_Taylor_long_term_planning;
    Tamara_Taylor_set_daytype_-_Tamara_Taylor --> Tamara_Taylor_perception_-_Tamara_Taylor_perceive_space;
    subgraph perception - Tamara Taylor
    Tamara_Taylor_perception_-_Tamara_Taylor_perceive_space --> Tamara_Taylor_perception_-_Tamara_Taylor_perceive_events;
    Tamara_Taylor_perception_-_Tamara_Taylor_perceive_events -.-> Tamara_Taylor_perception_-_Tamara_Taylor_store_events;
    end
    subgraph retrieval - Tamara Taylor
    Tamara_Taylor_retrieval_-_Tamara_Taylor_add_current_event --> Tamara_Taylor_retrieval_-_Tamara_Taylor_retrieve_events;
    Tamara_Taylor_retrieval_-_Tamara_Taylor_add_current_event --> Tamara_Taylor_retrieval_-_Tamara_Taylor_retrieve_thoughts;
    Tamara_Taylor_retrieval_-_Tamara_Taylor_retrieve_events --> Tamara_Taylor_retrieval_-_Tamara_Taylor___end__;
    Tamara_Taylor_retrieval_-_Tamara_Taylor_retrieve_thoughts --> Tamara_Taylor_retrieval_-_Tamara_Taylor___end__;
    end
    subgraph plan - Tamara Taylor
    Tamara_Taylor_plan_-_Tamara_Taylor_choose_retrieved --> Tamara_Taylor_plan_-_Tamara_Taylor_react;
    Tamara_Taylor_plan_-_Tamara_Taylor_determine_action --> Tamara_Taylor_plan_-_Tamara_Taylor_choose_retrieved;
    Tamara_Taylor_plan_-_Tamara_Taylor_long_term_planning --> Tamara_Taylor_plan_-_Tamara_Taylor_determine_action;
    Tamara_Taylor_plan_-_Tamara_Taylor_react --> Tamara_Taylor_plan_-_Tamara_Taylor_wrap_up;
    end
    subgraph reflection - Tamara Taylor
    Tamara_Taylor_reflection_-_Tamara_Taylor___start__ --> Tamara_Taylor_reflection_-_Tamara_Taylor_retrieve_last_conversation;
    Tamara_Taylor_reflection_-_Tamara_Taylor_reflect --> Tamara_Taylor_reflection_-_Tamara_Taylor___end__;
    Tamara_Taylor_reflection_-_Tamara_Taylor_reflect_on_conversation --> Tamara_Taylor_reflection_-_Tamara_Taylor___end__;
    Tamara_Taylor_reflection_-_Tamara_Taylor___start__ -. &nbsp;True&nbsp; .-> Tamara_Taylor_reflection_-_Tamara_Taylor_reflect;
    Tamara_Taylor_reflection_-_Tamara_Taylor___start__ -. &nbsp;False&nbsp; .-> Tamara_Taylor_reflection_-_Tamara_Taylor___end__;
    Tamara_Taylor_reflection_-_Tamara_Taylor_retrieve_last_conversation -. &nbsp;True&nbsp; .-> Tamara_Taylor_reflection_-_Tamara_Taylor_reflect_on_conversation;
    Tamara_Taylor_reflection_-_Tamara_Taylor_retrieve_last_conversation -. &nbsp;False&nbsp; .-> Tamara_Taylor_reflection_-_Tamara_Taylor___end__;
    end
    end
    classDef default fill:#f2f0ff,line-height:1.2
    classDef first fill-opacity:0
    classDef last fill:#bfb6fc
```