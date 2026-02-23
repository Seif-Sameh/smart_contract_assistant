"""Gradio UI for the Smart Contract Assistant."""

import requests
import gradio as gr

BACKEND_URL = "http://localhost:8000"


def upload_file(file) -> str:
    """Upload a document to the backend API."""
    if file is None:
        return "No file selected."

    try:
        with open(file.name, "rb") as f:
            filename = file.name.split("/")[-1]
            response = requests.post(
                f"{BACKEND_URL}/upload",
                files={"file": (filename, f)},
                timeout=60,
            )
        if response.status_code == 200:
            data = response.json()
            return f"✅ {data['message']} ({data['chunks_created']} chunks created)"
        else:
            return f"❌ Upload failed: {response.json().get('detail', response.text)}"
    except Exception as e:
        return f"❌ Error: {e}"


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
                    score = src.get("score", 0)
                    source_text += f"- Source {i}: {filename} (Page {page}, Score: {score:.2f})\n"
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
    if not filename.strip():
        return "Please enter a filename."

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
                    label="Upload PDF or DOCX",
                    file_types=[".pdf", ".docx"],
                )
            with gr.Row():
                upload_btn = gr.Button("Upload", variant="primary")
            with gr.Row():
                upload_status = gr.Textbox(label="Upload Status", interactive=False)

            upload_btn.click(fn=upload_file, inputs=[file_input], outputs=[upload_status])

            gr.Markdown("---")
            gr.Markdown("### 📝 Document Summary")
            with gr.Row():
                summary_filename = gr.Textbox(label="Filename to Summarize")
            with gr.Row():
                summary_btn = gr.Button("Get Summary")
            with gr.Row():
                summary_output = gr.Textbox(label="Summary", lines=10, interactive=False)

            summary_btn.click(fn=get_summary, inputs=[summary_filename], outputs=[summary_output])

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