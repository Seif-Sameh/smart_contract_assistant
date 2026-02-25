"""Gradio UI for the Smart Contract Assistant."""

import requests
import gradio as gr

BACKEND_URL = "http://localhost:8000"


def fetch_document_list():
    """Fetch the list of uploaded documents from the backend."""
    try:
        response = requests.get(f"{BACKEND_URL}/documents", timeout=10)
        if response.status_code == 200:
            return response.json().get("documents", [])
    except Exception:
        pass
    return []


def upload_file(files) -> str:
    """Upload one or more documents to the backend API."""
    if files is None or len(files) == 0:
        return "No files selected."

    try:
        opened_files = []
        try:
            for file in files:
                filename = file.name.split("/")[-1]
                opened_files.append(
                    ("files", (filename, open(file.name, "rb")))
                )

            response = requests.post(
                f"{BACKEND_URL}/upload",
                files=opened_files,
                timeout=120,
            )
        finally:
            # Close all file handles
            for _, (_, fh) in opened_files:
                fh.close()

        if response.status_code == 200:
            data = response.json()
            status_lines = [f"📁 {data['message']}"]
            for file_result in data.get("files", []):
                fname = file_result["filename"]
                fstatus = file_result["status"]
                if fstatus == "success":
                    chunks = file_result.get("chunks_created", 0)
                    status_lines.append(f"  ✅ {fname} — {chunks} chunks")
                elif fstatus == "skipped":
                    status_lines.append(f"  ⚠️ {fname} — {file_result.get('detail', 'skipped')}")
                else:
                    status_lines.append(f"  ❌ {fname} — {file_result.get('detail', 'error')}")
            return "\n".join(status_lines)
        else:
            return f"❌ Upload failed: {response.json().get('detail', response.text)}"
    except Exception as e:
        return f"❌ Error: {e}"


def refresh_document_dropdown():
    """Refresh the document dropdown with the latest uploaded files."""
    docs = fetch_document_list()
    return gr.Dropdown(choices=docs, value=None)


def on_upload_and_refresh(files):
    """Upload files, then refresh the dropdown."""
    status = upload_file(files)
    dropdown_update = refresh_document_dropdown()
    return status, dropdown_update


def chat(message: str, history: list, session_id: str):
    """Send a chat message and receive a response."""
    if not message.strip():
        return "", history, session_id

    try:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={"question": message, "session_id": session_id or None},
            timeout=60,
        )
        if response.status_code == 200:
            data = response.json()
            answer = data["answer"]

            # Format sources
            sources = data.get("sources", [])
            if sources:
                source_text = "\n\n📄 **Sources:**\n"
                for i, src in enumerate(sources, 1):
                    meta = src.get("metadata", {})
                    filename = meta.get("source", meta.get("filename", "Unknown"))
                    page = meta.get("page", "N/A")
                    score = src.get("rerank_score", src.get("score", 0))
                    source_text += f"- Source {i}: {filename} (Page {page}, Relevance: {score:.2f})\n"
                answer += source_text

            new_session_id = data.get("session_id", session_id)

            history = history + [
                gr.ChatMessage(role="user", content=message),
                gr.ChatMessage(role="assistant", content=answer),
            ]
            return "", history, new_session_id
        else:
            error = response.json().get("detail", response.text)
            history = history + [
                gr.ChatMessage(role="user", content=message),
                gr.ChatMessage(role="assistant", content=f"❌ Error: {error}"),
            ]
            return "", history, session_id
    except Exception as e:
        history = history + [
            gr.ChatMessage(role="user", content=message),
            gr.ChatMessage(role="assistant", content=f"❌ Connection error: {e}"),
        ]
        return "", history, session_id


def clear_chat():
    """Clear the chat history."""
    return "", [], ""


def get_summary(filename: str) -> str:
    """Request a document summary from the backend API."""
    if not filename or not filename.strip():
        return "Please select a document first."

    try:
        response = requests.post(
            f"{BACKEND_URL}/summarize",
            json={"filename": filename},
            timeout=120,
        )
        if response.status_code == 200:
            return response.json()["summary"]
        else:
            return f"❌ Error: {response.json().get('detail', response.text)}"
    except Exception as e:
        return f"❌ Connection error: {e}"


def build_ui() -> gr.Blocks:
    """Build and return the Gradio UI."""
    with gr.Blocks(title="Smart Contract Assistant") as demo:
        gr.Markdown("# 📄 Smart Contract Document Assistant")
        gr.Markdown("Upload contract documents and ask questions about them.")

        with gr.Tab("Upload Document"):
            with gr.Row():
                file_input = gr.File(
                    label="Upload PDF or DOCX files",
                    file_types=[".pdf", ".docx"],
                    file_count="multiple",
                )
            with gr.Row():
                upload_btn = gr.Button("Upload", variant="primary")
            with gr.Row():
                upload_status = gr.Textbox(label="Upload Status", interactive=False)

            gr.Markdown("---")
            gr.Markdown("### 📝 Document Summary")
            with gr.Row():
                summary_dropdown = gr.Dropdown(
                    label="Select Document to Summarize",
                    choices=fetch_document_list(),
                    interactive=True,
                )
                refresh_btn = gr.Button("🔄 Refresh", scale=0)
            with gr.Row():
                summary_btn = gr.Button("Get Summary")
            with gr.Row():
                summary_output = gr.Textbox(label="Summary", lines=10, interactive=False)

            # Upload → update status + refresh dropdown
            upload_btn.click(
                fn=on_upload_and_refresh,
                inputs=[file_input],
                outputs=[upload_status, summary_dropdown],
            )

            # Manual refresh button
            refresh_btn.click(
                fn=refresh_document_dropdown,
                outputs=[summary_dropdown],
            )

            # Summarize selected document
            summary_btn.click(
                fn=get_summary,
                inputs=[summary_dropdown],
                outputs=[summary_output],
            )

        with gr.Tab("Chat"):
            session_state = gr.State("")
            chatbot = gr.Chatbot(label="Conversation", height=400)
            with gr.Row():
                msg_box = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask a question about the uploaded document...",
                    scale=4,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear Chat")

            send_btn.click(
                fn=chat,
                inputs=[msg_box, chatbot, session_state],
                outputs=[msg_box, chatbot, session_state],
            )
            msg_box.submit(
                fn=chat,
                inputs=[msg_box, chatbot, session_state],
                outputs=[msg_box, chatbot, session_state],
            )
            clear_btn.click(
                fn=clear_chat,
                outputs=[msg_box, chatbot, session_state],
            )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860)