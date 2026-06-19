$(document).ready(function () {
    // Chuyển đổi giữa 2 form
    $('#link-show-register').click(function (e) {
        e.preventDefault();
        $('#form-login').addClass('d-none');
        $('#form-register').removeClass('d-none');
        $('#auth-subtitle').text('Create an account to sync your AI history');
    });

    $('#link-show-login').click(function (e) {
        e.preventDefault();
        $('#form-register').addClass('d-none');
        $('#form-login').removeClass('d-none');
        $('#auth-subtitle').text('Please sign in to access your workspace');
    });

    // Xử lý Đăng ký
    $('#form-register').submit(async function (e) {
        e.preventDefault();
        $('#register-error').addClass('d-none');
        $('#register-success').addClass('d-none');

        const payload = {
            username: $('#register-username').val(),
            password: $('#register-password').val()
        };

        const response = await fetch('/api/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        if (response.ok) {
            $('#register-success').text("Registration successful! Redirecting to Login...").removeClass('d-none');
            setTimeout(() => $('#link-show-login').click(), 2000);
        } else {
            $('#register-error').text(data.Message || "Registration failed. Check rules.").removeClass('d-none');
        }
    });

    $('#form-login').submit(async function (e) {
        e.preventDefault();
        $('#login-error').addClass('d-none');

        const payload = {
            username: $('#login-username').val(),
            password: $('#login-password').val()
        };

        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        if (response.ok) {
            localStorage.setItem('jwt_token', data.token);
            localStorage.setItem('username', data.username);
            window.location.href = '/';
        } else {
            $('#login-error').text("Invalid username or password.").removeClass('d-none');
        }
    });
});