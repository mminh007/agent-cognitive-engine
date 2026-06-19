let activeSessionId = "temp_guest_session";
const token = localStorage.getItem('jwt_token');
const savedUsername = localStorage.getItem('username');

$(document).ready(function () {

    if (token && savedUsername) {
        $('#lbl-username').text(savedUsername);
        $('#btn-auth-action').attr('title', 'Logout').html('<i class="bi bi-box-arrow-left fs-5 text-muted hover-danger"></i>');
        $('#guest-warning').addClass('d-none');
        loadChatSessions();
    } else {
        $('#lbl-username').text("Guest Account");
        $('#btn-auth-action').attr('title', 'Login/Register').html('<i class="bi bi-box-arrow-in-right fs-5 text-info"></i>');
        $('#guest-warning').removeClass('d-none');
        $('#session-container').html('<div class="text-center text-muted mt-5 small">Đăng nhập để lưu lịch sử</div>');
    }

    $('#btn-auth-action').click(function () {
        if (token) {
            localStorage.removeItem('jwt_token');
            localStorage.removeItem('username');
            window.location.reload();
        } else {
            window.location.href = '/Auth/Login';
        }
    });


    $('#user-input').on('input', function () {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        $('#btn-submit').prop('disabled', $.trim($(this).val()) === '');
    });

    $('#btn-submit').click(function () { executeChatStream(); });

    $('#user-input').keydown(function (e) {
        if (e.which === 13 && !e.shiftKey) {
            e.preventDefault();
            if (!$('#btn-submit').is(':disabled')) executeChatStream();
        }
    });
});


async function executeChatStream() {
    const promptInput = $('#user-input').val();
    if (!$.trim(promptInput)) return;

    $('#user-input').val('').css('height', 'auto');
    $('#btn-submit').prop('disabled', true);

    const messagesContainer = $('#chat-messages-container');
    messagesContainer.append(`<div class="message-box user">${promptInput}</div>`);
    messagesContainer.scrollTop(messagesContainer[0].scrollHeight);

    const aiMessageId = 'ai-msg-' + Date.now();
    messagesContainer.append(`
        <div class="message-box ai" id="${aiMessageId}">
            <i class="bi bi-robot me-2 text-info fs-5"></i><span class="tokens">...</span>
        </div>
    `);
    messagesContainer.scrollTop(messagesContainer[0].scrollHeight);
    const tokenTarget = $(`#${aiMessageId} .tokens`);

    const formData = new FormData();
    formData.append('prompt', promptInput);

    try {
        const headers = {};
        if (token) headers['Authorization'] = `Bearer ${token}`;

        const response = await fetch('/api/chat/stream', {
            method: 'POST', headers: headers, body: formData
        });

        if (response.status === 429) {
            tokenTarget.text("Bạn đang nhắn quá nhanh. Vui lòng thử lại sau."); return;
        }
        if (!response.ok) {
            tokenTarget.text("Lỗi kết nối Server."); return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        tokenTarget.text("");

        let aiAccumulatedText = "";

        // Optional: Configure marked to automatically convert \n to <br> tags cleanly
        marked.setOptions({
            breaks: true,
            gfm: true
        });

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const chunkText = decoder.decode(value);
            const lines = chunkText.split('\n');
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const actualToken = line.replace('data: ', '');
                    if (actualToken) {
                        // 1. Accumulate the new token into our global string variable
                        aiAccumulatedText += actualToken;

                        // 2. Parse the entire accumulated text into structural HTML elements
                        const parsedHtml = marked.parse(aiAccumulatedText);

                        // 3. Inject the compiled HTML directly into the UI container target
                        tokenTarget.html(parsedHtml);

                        // Keep the chat display strictly scrolled to the bottom during live emission
                        messagesContainer.scrollTop(messagesContainer[0].scrollHeight);
                    }
                }
            }
        }
    } catch (error) {
        tokenTarget.text("Mạch gRPC kết nối Core Engine bị gián đoạn.");
    }
}

function loadChatSessions() {
    if (!token) return;
    $.ajax({
        url: '/api/sessions/list', method: 'GET',
        headers: { 'Authorization': `Bearer ${token}` },
        success: function (data) {
            const container = $('#session-container');
            container.empty();
            if (!data || data.length === 0) {
                container.append('<div class="text-center text-muted mt-5 small">Chưa có cuộc trò chuyện nào</div>'); return;
            }
            data.forEach(session => {
                const isActive = session.id === activeSessionId ? 'active' : '';
                container.append(`
                    <a class="session-item ${isActive}" data-id="${session.id}" href="#">
                        <i class="bi bi-chat-left-text me-2 opacity-75"></i>
                        <span class="text-truncate" style="max-width: 190px;">${session.title}</span>
                    </a>
                `);
            });
        }
    });
}