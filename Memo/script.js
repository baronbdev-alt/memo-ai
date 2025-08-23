// Handle login
document.getElementById("loginForm").addEventListener("submit", function (e) {
    e.preventDefault(); // stop form reload

    let username = document.getElementById("username").value.trim();
    let password = document.getElementById("password").value.trim();

    if (username && password) {
        // ✅ Redirect to home page
        window.location.href = "home.html";
    } else {
        alert("Please enter username and password");
    }
});

