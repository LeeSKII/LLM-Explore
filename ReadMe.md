# LLM-Explore

测试各种 LLM 模型的集合之地，探索各种工程化的可能性。

## TODO List

- [ ] agno agent framework usage
  - [x] Agent Class - agent->agno-usage->concept->agent->agent.ipynb
  - [ ] Five levels of agentic systems
    - [x] Level 1: Agents with tools and instructions - agent->agno-usage->five-level->1_tools_instructions.ipynb
      - [x] Writing your own tools
        - [x] Stock agent
      - [ ] Exceptions
      - [ ] Hooks
      - [ ] Human in the loop
      - [ ] Tool kits
      - [ ] MCP
      - [ ] Writing your own toolkit
      - [ ] Selecting tools
      - [ ] Async tools
      - [ ] Tool Result Caching
    - [ ] Level 2: Agents with knowledge and storage
    - [ ] Level 3: Agents with memory and reasoning
    - [ ] Level 4: Teams of Agents with collaboration and coordination
    - [ ] Level 5: Multi-Agent Workflows with state and determinism
- [ ] web search agent use Tavily API
- [ ] deer flow deep researcher
- [ ] gpt deep researcher
- [ ] agno python agent framework
- [ ] sequential thinking [GitHub repository](https://github.com/FradSer/mcp-server-mas-sequential-thinking/blob/main/main.py)
- [ ] skyWork mock agentic seek deep researcher
- [ ] langGraph multi agent system
- [ ] chainlit cook book
- [ ] csv\pandas\sql agent
- [ ] langchain usage
- [ ] claude prompt
- [ ] chainlit weather agent, auto inject function description, and use light llm correct unstructured output format
- [ ] instructor prompt engineering
- [ ] MCP client
- [ ] MCP server

## Reflection

- Langchain 这样的框架可以用来提供快速化产品构建，整合资源是产品快速构建的**第一**手段
- 对于构建 Agent ，先从 prompt 开始，从 jupyter notebook 开始，逐步完善，最后整合到 .py 文件，再到 UI 界面
- 对话式的 UI 都可以优先使用 chainlit，然后是 streamlit，再然后是 taipy，最后是其他的 UI 框架
