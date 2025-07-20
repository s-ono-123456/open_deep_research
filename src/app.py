import streamlit as st
import asyncio
from open_deep_research.deep_researcher import deep_researcher
from open_deep_research.configuration import Configuration
from open_deep_research.state import AgentInputState
from open_deep_research.configuration import SearchAPI, MCPConfig, RunnableConfig

# deep_researcherを非同期で呼び出す関数
async def run_deep_researcher(question):

    input_state = AgentInputState(
        messages=[{"content": question, "type": "human"}],
    )

    # deep_researcherは非同期関数なのでawaitで呼び出します
    result = await deep_researcher.ainvoke(input_state, config.model_dump())
    # print("--- deep_researcher完了しました。 ---")
    # 最後のAIMessageを取得
    report = result.get("messages", [])[-1]
    report_content = report.content

    # 実行結果を返却
    return report_content

# ページタイトルと説明
st.set_page_config(page_title="DeepResearch", page_icon="🤖")
st.title("DeepResearch")
st.markdown("""
このアプリケーションは、質問に対して関連するWebページを検索し、情報を収集してレポートを作成します。
以下の質問を入力してください。
""")

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

# ユーザーからの質問入力
question = st.text_area("質問を入力してください", height=150)
# 入力がある場合のみ処理を実行
if question:
    # 処理中の表示
    with st.status("処理中...", expanded=True) as status:
        # 入力された質問を表示
        st.markdown("### 入力された質問")
        st.write(question)

        # ここでdeep_researcherを呼び出す処理を追加することができます。
        # 例えば、非同期関数を呼び出して結果を取得し、表示するなど。
        result = asyncio.run(run_deep_researcher(question))
    
    st.markdown("### 結果")
    st.write(result)
