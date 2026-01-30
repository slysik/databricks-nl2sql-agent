const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        Header, Footer, AlignmentType, LevelFormat, HeadingLevel, BorderStyle,
        WidthType, ShadingType, VerticalAlign, PageNumber, PageBreak,
        TableOfContents } = require('docx');
const fs = require('fs');

// Table border style
const tableBorder = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };

// Helper functions
const h1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
const h2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
const h3 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(text)] });
const p = (text) => new Paragraph({ spacing: { after: 120 }, children: [new TextRun(text)] });
const pBold = (text) => new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text, bold: true })] });
const code = (text) => new Paragraph({
  spacing: { after: 60, before: 60 },
  indent: { left: 360 },
  shading: { fill: "F5F5F5", type: ShadingType.CLEAR },
  children: [new TextRun({ text, font: "Courier New", size: 18 })]
});
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal",
        run: { size: 52, bold: true, color: "1F4E79", font: "Arial" },
        paragraph: { spacing: { before: 0, after: 200 }, alignment: AlignmentType.CENTER } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, color: "1F4E79", font: "Arial" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, color: "2E75B6", font: "Arial" },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, color: "404040", font: "Arial" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 } }
    ]
  },
  numbering: {
    config: [
      { reference: "bullet-main",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbered-steps",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbered-deploy",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbered-perf",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbered-sec",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "bullet-arch",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "bullet-security",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }
    ]
  },
  sections: [{
    properties: {
      page: { margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }
    },
    headers: {
      default: new Header({ children: [new Paragraph({
        alignment: AlignmentType.RIGHT,
        children: [new TextRun({ text: "Databricks NL2SQL Multi-Agent Cookbook", italics: true, size: 18, color: "666666" })]
      })] })
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Page ", size: 18 }), new TextRun({ children: [PageNumber.CURRENT], size: 18 }),
                   new TextRun({ text: " of ", size: 18 }), new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 18 }),
                   new TextRun({ text: "  |  Confidential - Ryder System, Inc.", size: 18, color: "666666" })]
      })] })
    },
    children: [
      // Title Page
      new Paragraph({ spacing: { before: 2400 } }),
      new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("Databricks NL2SQL")] }),
      new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("Multi-Agent System")] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 400, after: 400 },
        children: [new TextRun({ text: "Production Cookbook & Implementation Guide", size: 28, color: "666666" })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 800 },
        children: [new TextRun({ text: "Prepared for Ryder System, Inc.", size: 24 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200 },
        children: [new TextRun({ text: "January 2026", size: 22, color: "666666" })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1200 },
        children: [new TextRun({ text: "Version 1.0", size: 20, italics: true })] }),

      pageBreak(),

      // Table of Contents
      h1("Table of Contents"),
      new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" }),

      pageBreak(),

      // Executive Summary
      h1("1. Executive Summary"),
      p("This cookbook provides a comprehensive guide to the Databricks NL2SQL Multi-Agent System, designed for Ryder's supply chain and logistics AI initiatives. The system enables natural language querying of Delta tables and multi-agent orchestration for complex business automation."),
      new Paragraph({ spacing: { after: 120 }, children: [
        new TextRun({ text: "Key Capabilities:", bold: true })
      ]}),
      new Paragraph({ numbering: { reference: "bullet-main", level: 0 },
        children: [new TextRun("Natural Language to SQL conversion with defense-in-depth security")] }),
      new Paragraph({ numbering: { reference: "bullet-main", level: 0 },
        children: [new TextRun("Multi-agent orchestration using LangGraph supervisor pattern")] }),
      new Paragraph({ numbering: { reference: "bullet-main", level: 0 },
        children: [new TextRun("RAG integration via Databricks Vector Search")] }),
      new Paragraph({ numbering: { reference: "bullet-main", level: 0 },
        children: [new TextRun("Human-in-the-loop approval workflows")] }),
      new Paragraph({ numbering: { reference: "bullet-main", level: 0 },
        children: [new TextRun("MLflow deployment to Databricks Model Serving")] }),

      new Paragraph({ spacing: { before: 200, after: 120 }, children: [
        new TextRun({ text: "Target Use Cases for Ryder:", bold: true })
      ]}),
      new Paragraph({ numbering: { reference: "bullet-arch", level: 0 },
        children: [new TextRun("Demand Forecasting - ML on historical data for inventory optimization")] }),
      new Paragraph({ numbering: { reference: "bullet-arch", level: 0 },
        children: [new TextRun("Route Optimization - Real-time traffic, weather, delivery windows")] }),
      new Paragraph({ numbering: { reference: "bullet-arch", level: 0 },
        children: [new TextRun("Autonomous Procurement - Vendor selection with governance")] }),
      new Paragraph({ numbering: { reference: "bullet-arch", level: 0 },
        children: [new TextRun("Risk Management - Predictive analytics for supply chain disruptions")] }),

      pageBreak(),

      // Architecture Overview
      h1("2. Architecture Overview"),
      h2("2.1 System Architecture Diagram"),
      p("The following diagram illustrates the end-to-end flow from user query to response:"),

      // ASCII Architecture Diagram
      new Paragraph({ spacing: { before: 200, after: 200 }, alignment: AlignmentType.CENTER,
        shading: { fill: "F0F8FF", type: ShadingType.CLEAR },
        children: [new TextRun({ text: "ARCHITECTURE FLOW", bold: true, size: 20 })] }),
      code("                    +------------------+"),
      code("                    |   User Query     |"),
      code("                    | (Natural Lang)   |"),
      code("                    +--------+---------+"),
      code("                             |"),
      code("                             v"),
      code("                    +------------------+"),
      code("                    |  Streamlit UI    |"),
      code("                    |  (app/main.py)   |"),
      code("                    +--------+---------+"),
      code("                             |"),
      code("                             v"),
      code("     +-------------------+---+---+-------------------+"),
      code("     |                   |       |                   |"),
      code("     v                   v       v                   v"),
      code("+----------+    +----------+ +----------+    +----------+"),
      code("| NL2SQL   |    | Demand   | | Route    |    | Risk     |"),
      code("| Agent    |    | Forecast | | Optimize |    | Mgmt     |"),
      code("+----+-----+    +----+-----+ +----+-----+    +----+-----+"),
      code("     |              |            |               |"),
      code("     v              v            v               v"),
      code("+----------+  +----------+ +----------+   +----------+"),
      code("| SQL      |  | Vector   | | Vector   |   | Vector   |"),
      code("| Validator|  | Search   | | Search   |   | Search   |"),
      code("+----+-----+  +----+-----+ +----+-----+   +----+-----+"),
      code("     |              |            |               |"),
      code("     v              +-----+------+               |"),
      code("+----------+              |                      |"),
      code("| Delta    |              v                      v"),
      code("| Tables   |       +-------------+        +-------------+"),
      code("| (Unity)  |       |  Supervisor |        |  Human      |"),
      code("+----------+       |  Orchestr.  |        |  Approval   |"),
      code("                   +------+------+        +-------------+"),
      code("                          |"),
      code("                          v"),
      code("                   +-------------+"),
      code("                   | Synthesized |"),
      code("                   |  Response   |"),
      code("                   +-------------+"),

      new Paragraph({ spacing: { before: 300 } }),
      h2("2.2 Component Mapping (Azure vs Databricks)"),
      p("The following table shows the equivalence between Azure AI Foundry MACAE and Databricks Mosaic AI:"),

      new Table({
        columnWidths: [3120, 3120, 3120],
        rows: [
          new TableRow({ tableHeader: true, children: [
            new TableCell({ borders: cellBorders, width: { size: 3120, type: WidthType.DXA },
              shading: { fill: "1F4E79", type: ShadingType.CLEAR },
              children: [new Paragraph({ alignment: AlignmentType.CENTER,
                children: [new TextRun({ text: "Component", bold: true, color: "FFFFFF" })] })] }),
            new TableCell({ borders: cellBorders, width: { size: 3120, type: WidthType.DXA },
              shading: { fill: "1F4E79", type: ShadingType.CLEAR },
              children: [new Paragraph({ alignment: AlignmentType.CENTER,
                children: [new TextRun({ text: "Azure AI Foundry", bold: true, color: "FFFFFF" })] })] }),
            new TableCell({ borders: cellBorders, width: { size: 3120, type: WidthType.DXA },
              shading: { fill: "1F4E79", type: ShadingType.CLEAR },
              children: [new Paragraph({ alignment: AlignmentType.CENTER,
                children: [new TextRun({ text: "Databricks", bold: true, color: "FFFFFF" })] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, width: { size: 3120, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun("Agent Template")] })] }),
            new TableCell({ borders: cellBorders, width: { size: 3120, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun("FoundryAgentTemplate")] })] }),
            new TableCell({ borders: cellBorders, width: { size: 3120, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun("DatabricksAgentTemplate")] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, width: { size: 3120, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun("Orchestration")] })] }),
            new TableCell({ borders: cellBorders, width: { size: 3120, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun("MagenticBuilder")] })] }),
            new TableCell({ borders: cellBorders, width: { size: 3120, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun("StateGraph Supervisor")] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, width: { size: 3120, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun("RAG / Search")] })] }),
            new TableCell({ borders: cellBorders, width: { size: 3120, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun("Azure AI Search")] })] }),
            new TableCell({ borders: cellBorders, width: { size: 3120, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun("Vector Search")] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, width: { size: 3120, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun("Human Approval")] })] }),
            new TableCell({ borders: cellBorders, width: { size: 3120, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun("HumanApprovalManager")] })] }),
            new TableCell({ borders: cellBorders, width: { size: 3120, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun("interrupt() function")] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, width: { size: 3120, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun("State Persistence")] })] }),
            new TableCell({ borders: cellBorders, width: { size: 3120, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun("Cosmos DB")] })] }),
            new TableCell({ borders: cellBorders, width: { size: 3120, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun("Delta / Checkpointer")] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, width: { size: 3120, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun("Deployment")] })] }),
            new TableCell({ borders: cellBorders, width: { size: 3120, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun("Container Apps")] })] }),
            new TableCell({ borders: cellBorders, width: { size: 3120, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun("Model Serving")] })] })
          ]})
        ]
      }),

      pageBreak(),

      // Code Walkthrough
      h1("3. Code Walkthrough"),

      h2("3.1 NL2SQL Agent (sql_agent.py)"),
      p("The NL2SQL agent converts natural language to SQL queries against Delta tables. It uses LangChain's SQL toolkit with ChatDatabricks."),

      pBold("Key Code Pattern:"),
      code("class DatabricksNL2SQLAgent:"),
      code("    def __init__(self, catalog: str, schema: str,"),
      code("                 model_endpoint: str = 'databricks-meta-llama-3-3-70b-instruct'):"),
      code("        # SQLDatabase handles connection pooling, Unity Catalog auth"),
      code("        self.db = SQLDatabase.from_databricks(catalog=catalog, schema=schema)"),
      code("        "),
      code("        # Temperature=0 for deterministic SQL generation"),
      code("        self.llm = ChatDatabricks(endpoint=model_endpoint, temperature=0.0)"),
      code("        "),
      code("        # Build agent with SQL tools"),
      code("        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)"),
      code("        self.agent = create_react_agent("),
      code("            model=self.llm,"),
      code("            tools=toolkit.get_tools(),"),
      code("            state_modifier=system_prompt  # Security rules here"),
      code("        )"),

      new Paragraph({ spacing: { before: 200 } }),
      pBold("Usage Example:"),
      code("agent = DatabricksNL2SQLAgent('ryder_chatbot', 'agent_accessible')"),
      code("result = agent.query('Which vendors have reliability > 95%?')"),
      code("print(result['answer'])  # Natural language response with data"),

      h2("3.2 Base Agent Template (base_agent.py)"),
      p("The DatabricksAgentTemplate provides a reusable pattern equivalent to Azure's FoundryAgentTemplate. It supports RAG tools and implements the MLflow ResponsesAgent interface for deployment."),

      pBold("LangGraph Agent Pattern:"),
      code("class DatabricksAgentTemplate(ResponsesAgent):"),
      code("    def _build_graph(self):"),
      code("        # LLM initialization"),
      code("        self.llm = ChatDatabricks(endpoint=self.model_endpoint)"),
      code("        self.llm_with_tools = self.llm.bind_tools(self.tools)"),
      code("        "),
      code("        # Build StateGraph (equivalent to ChatAgent)"),
      code("        builder = StateGraph(AgentState)"),
      code("        builder.add_node('agent', agent_node)"),
      code("        builder.add_node('tools', ToolNode(tools=self.tools))"),
      code("        builder.add_conditional_edges('agent', should_continue,"),
      code("                                      {'tools': 'tools', 'end': END})"),
      code("        builder.add_edge('tools', 'agent')"),
      code("        builder.add_edge(START, 'agent')"),
      code("        self._graph = builder.compile()"),

      h2("3.3 Supervisor Orchestration (supervisor.py)"),
      p("The supervisor pattern coordinates multiple specialized agents. It decides which agent to call based on the task and synthesizes final results."),

      pBold("Orchestration Flow:"),
      code("Supervisor -> Agent1 -> Supervisor -> Agent2 -> ... -> Synthesize"),

      new Paragraph({ spacing: { before: 120 } }),
      pBold("Key Implementation:"),
      code("class DatabricksOrchestrationManager:"),
      code("    def _build_supervisor_graph(self):"),
      code("        supervisor_prompt = f'''You orchestrate: {agent_names}"),
      code("        Decide which agent to call next. Respond with agent name or FINISH.'''"),
      code("        "),
      code("        builder = StateGraph(OrchestratorState)"),
      code("        builder.add_node('supervisor', supervisor_node)"),
      code("        for name in agent_names:"),
      code("            builder.add_node(name, create_agent_node(name))"),
      code("            builder.add_edge(name, 'supervisor')  # Return to supervisor"),
      code("        builder.add_node('synthesize', synthesize_node)"),
      code("        builder.add_conditional_edges('supervisor', route_to_agent)"),
      code("        self.workflow = builder.compile(checkpointer=self.checkpointer)"),

      h2("3.4 Human Approval (human_approval.py)"),
      p("Human-in-the-loop approval uses LangGraph's interrupt() function to pause execution for review."),

      pBold("Approval Tool Pattern:"),
      code("@tool"),
      code("def request_human_approval(action: str, details: str, risk_level: str) -> str:"),
      code("    '''Request approval before high-stakes actions.'''"),
      code("    approval_request = {"),
      code("        'action': action,"),
      code("        'details': details,"),
      code("        'risk_level': risk_level"),
      code("    }"),
      code("    # Pause execution for human review"),
      code("    response = interrupt(approval_request)"),
      code("    if response.get('approved'):"),
      code("        return 'APPROVED. Proceed with action.'"),
      code("    return f'REJECTED: {response.get(\"reason\")}'"),

      pageBreak(),

      // End-to-End Flow
      h1("4. End-to-End Processing Flow"),

      h2("4.1 NL2SQL Query Flow"),
      new Paragraph({ numbering: { reference: "numbered-steps", level: 0 },
        children: [new TextRun({ text: "User Input: ", bold: true }), new TextRun("User enters natural language query in Streamlit UI")] }),
      new Paragraph({ numbering: { reference: "numbered-steps", level: 0 },
        children: [new TextRun({ text: "Agent Initialization: ", bold: true }), new TextRun("DatabricksNL2SQLAgent connects to Unity Catalog")] }),
      new Paragraph({ numbering: { reference: "numbered-steps", level: 0 },
        children: [new TextRun({ text: "Schema Introspection: ", bold: true }), new TextRun("SQLDatabaseToolkit retrieves table schemas")] }),
      new Paragraph({ numbering: { reference: "numbered-steps", level: 0 },
        children: [new TextRun({ text: "SQL Generation: ", bold: true }), new TextRun("LLM generates SQL based on schema + system prompt")] }),
      new Paragraph({ numbering: { reference: "numbered-steps", level: 0 },
        children: [new TextRun({ text: "Security Validation: ", bold: true }), new TextRun("SQL Validator (AST parsing) checks for dangerous patterns")] }),
      new Paragraph({ numbering: { reference: "numbered-steps", level: 0 },
        children: [new TextRun({ text: "Query Execution: ", bold: true }), new TextRun("Validated SQL runs against Delta tables via Unity Catalog RBAC")] }),
      new Paragraph({ numbering: { reference: "numbered-steps", level: 0 },
        children: [new TextRun({ text: "Response Generation: ", bold: true }), new TextRun("LLM formats results into natural language answer")] }),

      h2("4.2 Multi-Agent Orchestration Flow"),
      new Paragraph({ numbering: { reference: "numbered-deploy", level: 0 },
        children: [new TextRun({ text: "Task Submission: ", bold: true }), new TextRun("Complex task submitted to orchestrator")] }),
      new Paragraph({ numbering: { reference: "numbered-deploy", level: 0 },
        children: [new TextRun({ text: "Supervisor Analysis: ", bold: true }), new TextRun("Supervisor LLM determines which agent to call")] }),
      new Paragraph({ numbering: { reference: "numbered-deploy", level: 0 },
        children: [new TextRun({ text: "Agent Execution: ", bold: true }), new TextRun("Selected agent processes task with RAG tools")] }),
      new Paragraph({ numbering: { reference: "numbered-deploy", level: 0 },
        children: [new TextRun({ text: "Return to Supervisor: ", bold: true }), new TextRun("Agent response added to state, supervisor decides next")] }),
      new Paragraph({ numbering: { reference: "numbered-deploy", level: 0 },
        children: [new TextRun({ text: "Approval Check: ", bold: true }), new TextRun("If high-stakes decision, interrupt() pauses for approval")] }),
      new Paragraph({ numbering: { reference: "numbered-deploy", level: 0 },
        children: [new TextRun({ text: "Iteration: ", bold: true }), new TextRun("Continue until supervisor returns FINISH or max_rounds reached")] }),
      new Paragraph({ numbering: { reference: "numbered-deploy", level: 0 },
        children: [new TextRun({ text: "Synthesis: ", bold: true }), new TextRun("All agent responses synthesized into unified output")] }),

      pageBreak(),

      // Production Deployment
      h1("5. Production Deployment Guide"),

      h2("5.1 Prerequisites"),
      new Paragraph({ numbering: { reference: "bullet-security", level: 0 },
        children: [new TextRun("Databricks workspace with Unity Catalog enabled")] }),
      new Paragraph({ numbering: { reference: "bullet-security", level: 0 },
        children: [new TextRun("Model Serving endpoint (e.g., databricks-meta-llama-3-3-70b-instruct)")] }),
      new Paragraph({ numbering: { reference: "bullet-security", level: 0 },
        children: [new TextRun("Vector Search endpoints for RAG (if using multi-agent system)")] }),
      new Paragraph({ numbering: { reference: "bullet-security", level: 0 },
        children: [new TextRun("Service principal with appropriate permissions")] }),

      h2("5.2 Deployment Steps"),

      h3("Step 1: Set Up Unity Catalog"),
      code("-- Create catalog and schema"),
      code("CREATE CATALOG IF NOT EXISTS ryder_chatbot;"),
      code("USE CATALOG ryder_chatbot;"),
      code("CREATE SCHEMA IF NOT EXISTS agent_accessible;"),
      code(""),
      code("-- Create tables (see data/schema.sql for full schema)"),
      code("CREATE TABLE vendors (...) USING DELTA;"),
      code("CREATE TABLE routes (...) USING DELTA;"),

      h3("Step 2: Configure RBAC"),
      code("-- Create service principal with read-only access"),
      code("GRANT USE CATALOG ON CATALOG ryder_chatbot TO `nl2sql-agent-sp`;"),
      code("GRANT USE SCHEMA ON SCHEMA ryder_chatbot.agent_accessible TO `nl2sql-agent-sp`;"),
      code("GRANT SELECT ON SCHEMA ryder_chatbot.agent_accessible TO `nl2sql-agent-sp`;"),

      h3("Step 3: Deploy with MLflow"),
      code("import mlflow"),
      code(""),
      code("with mlflow.start_run():"),
      code("    mlflow.pyfunc.log_model("),
      code("        artifact_path='nl2sql_agent',"),
      code("        python_model=agent,"),
      code("        registered_model_name='ryder_nl2sql_agent',"),
      code("        pip_requirements=["),
      code("            'langgraph>=1.0.5',"),
      code("            'langchain-databricks',"),
      code("            'sqlparse>=0.5.0'"),
      code("        ]"),
      code("    )"),

      h3("Step 4: Create Model Serving Endpoint"),
      p("Use the Databricks UI or API to deploy the registered model:"),
      code("# Via API"),
      code("from databricks.sdk import WorkspaceClient"),
      code(""),
      code("w = WorkspaceClient()"),
      code("w.serving_endpoints.create("),
      code("    name='ryder-nl2sql-endpoint',"),
      code("    config={"),
      code("        'served_models': [{"),
      code("            'model_name': 'ryder_nl2sql_agent',"),
      code("            'model_version': '1',"),
      code("            'workload_size': 'Small',"),
      code("            'scale_to_zero_enabled': True"),
      code("        }]"),
      code("    }"),
      code(")"),

      h2("5.3 Environment Configuration"),
      new Table({
        columnWidths: [3500, 5860],
        rows: [
          new TableRow({ tableHeader: true, children: [
            new TableCell({ borders: cellBorders, width: { size: 3500, type: WidthType.DXA },
              shading: { fill: "1F4E79", type: ShadingType.CLEAR },
              children: [new Paragraph({ children: [new TextRun({ text: "Variable", bold: true, color: "FFFFFF" })] })] }),
            new TableCell({ borders: cellBorders, width: { size: 5860, type: WidthType.DXA },
              shading: { fill: "1F4E79", type: ShadingType.CLEAR },
              children: [new Paragraph({ children: [new TextRun({ text: "Description", bold: true, color: "FFFFFF" })] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("DATABRICKS_HOST")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Workspace URL (e.g., https://adb-xxx.azuredatabricks.net)")] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("DATABRICKS_TOKEN")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Personal access token or service principal token")] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("MODEL_ENDPOINT")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("LLM endpoint (default: databricks-meta-llama-3-3-70b-instruct)")] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("CATALOG_NAME")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Unity Catalog name (e.g., ryder_chatbot)")] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("SCHEMA_NAME")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Schema name (e.g., agent_accessible)")] })] })
          ]})
        ]
      }),

      pageBreak(),

      // Performance Considerations
      h1("6. Performance Optimization"),

      h2("6.1 Key Performance Recommendations"),

      new Paragraph({ numbering: { reference: "numbered-perf", level: 0 }, spacing: { after: 80 },
        children: [new TextRun({ text: "LLM Temperature: ", bold: true }), new TextRun("Set temperature=0.0 for SQL generation to ensure deterministic, consistent queries.")] }),

      new Paragraph({ numbering: { reference: "numbered-perf", level: 0 }, spacing: { after: 80 },
        children: [new TextRun({ text: "Connection Pooling: ", bold: true }), new TextRun("SQLDatabase.from_databricks() handles connection pooling automatically. Cache agent instances with @st.cache_resource.")] }),

      new Paragraph({ numbering: { reference: "numbered-perf", level: 0 }, spacing: { after: 80 },
        children: [new TextRun({ text: "Vector Search Optimization: ", bold: true }), new TextRun("Limit num_results (default: 5-10) to reduce token usage while maintaining relevance.")] }),

      new Paragraph({ numbering: { reference: "numbered-perf", level: 0 }, spacing: { after: 80 },
        children: [new TextRun({ text: "Orchestration Limits: ", bold: true }), new TextRun("Set max_rounds (default: 10) to prevent infinite loops. Monitor round_count in state.")] }),

      new Paragraph({ numbering: { reference: "numbered-perf", level: 0 }, spacing: { after: 80 },
        children: [new TextRun({ text: "Checkpointing: ", bold: true }), new TextRun("Use InMemorySaver for development; switch to Delta-based persistence for production state recovery.")] }),

      new Paragraph({ numbering: { reference: "numbered-perf", level: 0 }, spacing: { after: 80 },
        children: [new TextRun({ text: "Model Serving Scale: ", bold: true }), new TextRun("Enable scale_to_zero for dev/test. Use dedicated endpoints with autoscaling for production workloads.")] }),

      h2("6.2 Performance Metrics to Monitor"),
      new Table({
        columnWidths: [2800, 3280, 3280],
        rows: [
          new TableRow({ tableHeader: true, children: [
            new TableCell({ borders: cellBorders, width: { size: 2800, type: WidthType.DXA },
              shading: { fill: "1F4E79", type: ShadingType.CLEAR },
              children: [new Paragraph({ children: [new TextRun({ text: "Metric", bold: true, color: "FFFFFF" })] })] }),
            new TableCell({ borders: cellBorders, width: { size: 3280, type: WidthType.DXA },
              shading: { fill: "1F4E79", type: ShadingType.CLEAR },
              children: [new Paragraph({ children: [new TextRun({ text: "Target", bold: true, color: "FFFFFF" })] })] }),
            new TableCell({ borders: cellBorders, width: { size: 3280, type: WidthType.DXA },
              shading: { fill: "1F4E79", type: ShadingType.CLEAR },
              children: [new Paragraph({ children: [new TextRun({ text: "Action if Exceeded", bold: true, color: "FFFFFF" })] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Query Latency")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("< 5 seconds (P95)")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Scale endpoint, optimize SQL")] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Token Usage")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("< 4K tokens/request")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Reduce RAG results, trim prompts")] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Orchestration Rounds")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("< 5 rounds average")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Improve supervisor prompt")] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("SQL Validation Failures")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("< 5% of queries")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Enhance system prompt rules")] })] })
          ]})
        ]
      }),

      pageBreak(),

      // Security Considerations
      h1("7. Security Considerations"),

      h2("7.1 Defense-in-Depth Architecture"),
      p("The system implements three layers of security to protect against SQL injection and data exfiltration:"),

      new Paragraph({ spacing: { before: 160 }, children: [new TextRun({ text: "Layer 1: Unity Catalog RBAC", bold: true, color: "C00000" })] }),
      new Paragraph({ numbering: { reference: "bullet-main", level: 0 },
        children: [new TextRun("Service principal has SELECT-only permissions on approved tables")] }),
      new Paragraph({ numbering: { reference: "bullet-main", level: 0 },
        children: [new TextRun("Row-level and column-level security enforced at catalog level")] }),
      new Paragraph({ numbering: { reference: "bullet-main", level: 0 },
        children: [new TextRun("Audit logging of all queries via system tables")] }),

      new Paragraph({ spacing: { before: 160 }, children: [new TextRun({ text: "Layer 2: Hardened System Prompt", bold: true, color: "C00000" })] }),
      new Paragraph({ numbering: { reference: "bullet-arch", level: 0 },
        children: [new TextRun("Explicit rules: 'Generate ONLY SELECT statements'")] }),
      new Paragraph({ numbering: { reference: "bullet-arch", level: 0 },
        children: [new TextRun("Scope restriction: 'Query ONLY tables in {catalog}.{schema}'")] }),
      new Paragraph({ numbering: { reference: "bullet-arch", level: 0 },
        children: [new TextRun("Result limits: 'Use LIMIT to restrict results (default: 100)'")] }),

      new Paragraph({ spacing: { before: 160 }, children: [new TextRun({ text: "Layer 3: SQL Validator (AST Parsing)", bold: true, color: "C00000" })] }),
      new Paragraph({ numbering: { reference: "bullet-security", level: 0 },
        children: [new TextRun("Parses SQL into AST via sqlparse library")] }),
      new Paragraph({ numbering: { reference: "bullet-security", level: 0 },
        children: [new TextRun("Blocks dangerous keywords: DROP, DELETE, UPDATE, INSERT, ALTER, UNION, --")] }),
      new Paragraph({ numbering: { reference: "bullet-security", level: 0 },
        children: [new TextRun("Enforces single-statement execution (prevents chained attacks)")] }),
      new Paragraph({ numbering: { reference: "bullet-security", level: 0 },
        children: [new TextRun("Auto-injects LIMIT clause if missing")] }),

      h2("7.2 SQL Validator Implementation"),
      code("BLOCKED_KEYWORDS = frozenset(["),
      code("    'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE',"),
      code("    'TRUNCATE', 'GRANT', 'REVOKE', 'UNION', '--', ';'"),
      code("])"),
      code(""),
      code("def validate_sql(sql: str) -> str:"),
      code("    parsed = sqlparse.parse(sql)"),
      code("    "),
      code("    if len(parsed) != 1:"),
      code("        raise SecurityError('Multiple statements blocked')"),
      code("    "),
      code("    if parsed[0].get_type() != 'SELECT':"),
      code("        raise SecurityError(f'Only SELECT allowed. Got: {parsed[0].get_type()}')"),
      code("    "),
      code("    sql_upper = sql.upper()"),
      code("    for keyword in BLOCKED_KEYWORDS:"),
      code("        if keyword in sql_upper:"),
      code("            raise SecurityError(f'Blocked keyword: {keyword}')"),
      code("    "),
      code("    if 'LIMIT' not in sql_upper:"),
      code("        sql = f'{sql} LIMIT 1000'"),
      code("    return sql"),

      h2("7.3 Production Security Checklist"),
      new Table({
        columnWidths: [4680, 4680],
        rows: [
          new TableRow({ tableHeader: true, children: [
            new TableCell({ borders: cellBorders, width: { size: 4680, type: WidthType.DXA },
              shading: { fill: "1F4E79", type: ShadingType.CLEAR },
              children: [new Paragraph({ children: [new TextRun({ text: "Requirement", bold: true, color: "FFFFFF" })] })] }),
            new TableCell({ borders: cellBorders, width: { size: 4680, type: WidthType.DXA },
              shading: { fill: "1F4E79", type: ShadingType.CLEAR },
              children: [new Paragraph({ children: [new TextRun({ text: "Implementation", bold: true, color: "FFFFFF" })] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Authentication")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Service principal with minimal permissions")] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Authorization")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Unity Catalog RBAC with SELECT-only grants")] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Input Validation")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("SQL validator with AST parsing")] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Output Sanitization")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("LIMIT enforcement, PII masking")] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Audit Logging")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Unity Catalog system tables + MLflow tracking")] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Network Security")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Private Link / VNet injection")] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Secrets Management")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Databricks Secrets or Azure Key Vault")] })] })
          ]})
        ]
      }),

      h2("7.4 Human Approval Governance"),
      p("For autonomous decision-making (vendor selection, route changes), the system implements approval thresholds:"),

      new Table({
        columnWidths: [3120, 3120, 3120],
        rows: [
          new TableRow({ tableHeader: true, children: [
            new TableCell({ borders: cellBorders, width: { size: 3120, type: WidthType.DXA },
              shading: { fill: "1F4E79", type: ShadingType.CLEAR },
              children: [new Paragraph({ children: [new TextRun({ text: "Decision Type", bold: true, color: "FFFFFF" })] })] }),
            new TableCell({ borders: cellBorders, width: { size: 3120, type: WidthType.DXA },
              shading: { fill: "1F4E79", type: ShadingType.CLEAR },
              children: [new Paragraph({ children: [new TextRun({ text: "Threshold", bold: true, color: "FFFFFF" })] })] }),
            new TableCell({ borders: cellBorders, width: { size: 3120, type: WidthType.DXA },
              shading: { fill: "1F4E79", type: ShadingType.CLEAR },
              children: [new Paragraph({ children: [new TextRun({ text: "Escalation", bold: true, color: "FFFFFF" })] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Vendor Selection")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("> $100,000 contract")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Senior Management")] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Vendor Selection")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Risk score > 0.7")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Risk Committee")] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Route Change")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("> 20% deviation")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Operations Lead")] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Route Change")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("SLA impact possible")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Customer Success")] })] })
          ]})
        ]
      }),

      pageBreak(),

      // Appendix
      h1("8. Appendix"),

      h2("8.1 File Structure"),
      code("databricks_prototype/"),
      code("  agents/"),
      code("    base_agent.py        # DatabricksAgentTemplate (ResponsesAgent)"),
      code("    sql_agent.py         # NL2SQL agent implementation"),
      code("    supply_chain_agents.py  # Ryder-specific agent team"),
      code("  orchestration/"),
      code("    supervisor.py        # Multi-agent orchestration"),
      code("    human_approval.py    # interrupt() approval workflow"),
      code("  tools/"),
      code("    rag_tools.py         # Vector Search RAG tools"),
      code("  security/"),
      code("    sql_validator.py     # AST-based SQL validation"),
      code("  app/"),
      code("    main.py              # Streamlit chat UI"),
      code("  data/"),
      code("    schema.sql           # Delta table definitions"),
      code("  notebooks/"),
      code("    01_supply_chain_demo.py  # End-to-end demonstration"),
      code("  requirements.txt       # Python dependencies"),

      h2("8.2 Quick Start Commands"),
      code("# Install dependencies"),
      code("pip install -r databricks_prototype/requirements.txt"),
      code(""),
      code("# Run Streamlit app locally"),
      code("streamlit run databricks_prototype/app/main.py"),
      code(""),
      code("# Run demo notebook in Databricks"),
      code("# Import notebooks/01_supply_chain_demo.py to workspace"),

      h2("8.3 Sample Queries"),
      new Table({
        columnWidths: [4680, 4680],
        rows: [
          new TableRow({ tableHeader: true, children: [
            new TableCell({ borders: cellBorders, width: { size: 4680, type: WidthType.DXA },
              shading: { fill: "1F4E79", type: ShadingType.CLEAR },
              children: [new Paragraph({ children: [new TextRun({ text: "Natural Language", bold: true, color: "FFFFFF" })] })] }),
            new TableCell({ borders: cellBorders, width: { size: 4680, type: WidthType.DXA },
              shading: { fill: "1F4E79", type: ShadingType.CLEAR },
              children: [new Paragraph({ children: [new TextRun({ text: "Generated SQL", bold: true, color: "FFFFFF" })] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Which vendors have reliability > 95%?")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun({ text: "SELECT * FROM vendors WHERE reliability_pct > 95", font: "Courier New", size: 18 })] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Show routes from Phoenix hub")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun({ text: "SELECT * FROM routes WHERE origin_hub = 'Phoenix'", font: "Courier New", size: 18 })] })] })
          ]}),
          new TableRow({ children: [
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun("Count questions by topic")] })] }),
            new TableCell({ borders: cellBorders, children: [new Paragraph({ children: [new TextRun({ text: "SELECT topic, COUNT(*) FROM customer_questions GROUP BY topic", font: "Courier New", size: 18 })] })] })
          ]})
        ]
      }),

      new Paragraph({ spacing: { before: 600 } }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200 },
        children: [new TextRun({ text: "--- End of Document ---", italics: true, color: "666666" })] })
    ]
  }]
});

// Create docs directory if it doesn't exist
const docsDir = '/Users/slysik/foundry/Multi-Agent-Custom-Automation-Engine-Solution-Accelerator/databricks_prototype/docs';
if (!fs.existsSync(docsDir)) {
  fs.mkdirSync(docsDir, { recursive: true });
}

// Generate document
Packer.toBuffer(doc).then(buffer => {
  const outputPath = `${docsDir}/Databricks_NL2SQL_Cookbook.docx`;
  fs.writeFileSync(outputPath, buffer);
  console.log(`Document created: ${outputPath}`);
});
