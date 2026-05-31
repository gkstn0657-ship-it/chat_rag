// script_v4.js - RAG 통합 채팅 + 업로드 기능 통합

// ============================================
// 채팅 기능 (/integrated-chat)
// ============================================

function sendMessage() {
    const inputElement = document.getElementById("user-input");
    const userMessage = inputElement.value;

    if (userMessage === "") return;

    addMessage("user", userMessage);
    inputElement.value = "";

    fetch("http://localhost:8000/integrated-chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: userMessage })
    })
    .then(function(response) {
        return response.json().then(function(data) {
            return { ok: response.ok, status: response.status, data: data };
        });
    })
    .then(function(result) {
        if (!result.ok) {
            var errMsg = result.data.detail || "서버 오류 (HTTP " + result.status + ")";
            addMessage("bot", "오류: " + errMsg);
            return;
        }
        addMessage("bot", result.data.answer, result.data.source);
    })
    .catch(function(error) {
        addMessage("bot", "서버 연결에 실패했습니다: " + error.message);
    });
}

function addMessage(type, text, source) {
    const chatWindow = document.getElementById("chat-window");
    const messageDiv = document.createElement("div");
    messageDiv.className = "message " + type;
    messageDiv.innerText = text;

    if (type === "bot" && source) {
        const sourceDiv = document.createElement("div");
        sourceDiv.className = "source-tag";
        sourceDiv.innerText = "출처: " + source;
        messageDiv.appendChild(sourceDiv);
    }

    chatWindow.appendChild(messageDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

// Enter 키로 메시지 전송
document.getElementById("user-input").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});

// ============================================
// 업로드 패널 열기/닫기
// ============================================

function toggleUpload() {
    const panel = document.getElementById("upload-panel");
    panel.classList.toggle("hidden");
}

// ============================================
// 파일 업로드 (AJAX - form 제출 없음)
// ============================================

document.getElementById("upload-btn").addEventListener("click", async function() {
    const fileInput = document.getElementById("file-input");
    const file = fileInput.files[0];

    if (!file) {
        addMessage("bot", "파일을 먼저 선택해주세요.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    addMessage("bot", "'" + file.name + "' 업로드 중...");

    try {
        const response = await fetch("http://localhost:8000/upload", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const errData = await response.json().catch(function() { return null; });
            const detail = errData && errData.detail ? errData.detail : "서버 오류 (HTTP " + response.status + ")";
            addMessage("bot", "업로드 실패: " + detail);
            return;
        }

        const data = await response.json();
        console.log("업로드 성공:", data);
        addMessage("bot", data.message);

        // 업로드 완료 후 파일 입력 초기화
        fileInput.value = "";

    } catch(error) {
        console.error("업로드 에러:", error);
        addMessage("bot", "서버 연결 실패: " + error.message);
    }
});
