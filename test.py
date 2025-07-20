"""
このスクリプトは、src/open_deep_research/deep_researcher.pyのdeep_researcherを呼び出してテストします。
"""
# 必要なモジュールをインポート
import asyncio
from src.open_deep_research.deep_researcher import deep_researcher
from src.open_deep_research.configuration import Configuration
from src.open_deep_research.state import AgentInputState
from src.open_deep_research.configuration import SearchAPI, MCPConfig, RunnableConfig

# テスト用の設定と入力状態を作成
config = Configuration(
    # General Configuration
    max_structured_output_retries=1,
    allow_clarification=True,
    max_concurrent_research_units=1,
    # Research Configuration
    search_api=SearchAPI.TAVILY,
    max_researcher_iterations=2,
    max_react_tool_calls=2,
    # Model Configuration
    summarization_model="openai:gpt-4.1-mini",
    summarization_model_max_tokens=8192,
    research_model="openai:gpt-4.1-nano",  # モデル名は適宜変更してください
    research_model_max_tokens=10000,
    compression_model="openai:gpt-4.1-mini",
    compression_model_max_tokens=8192,
    final_report_model="openai:gpt-4.1",
    final_report_model_max_tokens=10000,
    # MCP server configuration
    # mcp_config=None,
    # mcp_prompt=None,
)

input_state = AgentInputState(
    messages=[{"content": """
               mcp-server-filesystemを利用して、open_deep_researchのファイルを確認してください。
               その後、このファイルで用意されているDeepResearch処理ではOpenAIのgpt-4.1などのモデルを利用していますが、
               AIモデルとしてAzure OpenAI ServicesのAPIを利用するにはソースコードのどこを変えればよいのかの調査を行い、結果をまとめてください。
               """, "type": "human"}],
)

# deep_researcherを非同期で呼び出す関数
async def run_deep_researcher():
    # deep_researcherは非同期関数なのでawaitで呼び出します
    result = await deep_researcher.ainvoke(input_state)
    print("--- deep_researcher完了しました。 ---")
    # 最後のAIMessageを取得
    report = result.get("messages", [])[-1]
    report_content = report.content
    # 実行結果をファイルに出力
    with open("output.md", "w") as f:
        f.write(report_content)

# メイン処理
if __name__ == "__main__":
    # 非同期関数を実行
    asyncio.run(run_deep_researcher())
