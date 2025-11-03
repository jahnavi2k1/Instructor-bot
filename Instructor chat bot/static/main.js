// Chat history: [{role: "user"|"model", text: "..."}]
let chatHistory = [];
let currentMode = "all_subjects";
let isProcessing = false;
let currentTheme = localStorage.getItem("theme") || "pink";

// API base (ensure we hit the Flask server; avoid file:// origin)
const ORIGIN = window.location.origin || "";
const API_BASE = (ORIGIN && ORIGIN.startsWith("http")) ? ORIGIN : "http://localhost:8000";

// DOM Elements
const chatMessages = document.getElementById("chat-messages");
const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");
const modeSelect = document.getElementById("mode-select");
const statusText = document.getElementById("status-text");
const clearButton = document.getElementById("clear-button");
const dropdownButton = document.getElementById("dropdown-button");
const dropdownMenu = document.getElementById("dropdown-menu");
const dropdownItems = dropdownMenu ? dropdownMenu.querySelectorAll(".dropdown-item") : [];
const dropdownText = dropdownButton ? dropdownButton.querySelector(".dropdown-text") : null;

// Theme Elements
const themeButton = document.getElementById("theme-button");
const themeMenu = document.getElementById("theme-menu");
const themeItems = themeMenu ? themeMenu.querySelectorAll(".theme-item") : [];

// Apply theme on load
function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    currentTheme = theme;
    localStorage.setItem("theme", theme);

    // Update selected theme item
    themeItems.forEach(item => {
        item.classList.remove("selected");
        if (item.getAttribute("data-theme") === theme) {
            item.classList.add("selected");
        }
    });
}

// Initialize theme
applyTheme(currentTheme);

// Theme button click handler
if (themeButton && themeMenu) {
    themeButton.addEventListener("click", function (e) {
        e.stopPropagation();
        e.preventDefault();
        themeMenu.classList.toggle("show");
        themeButton.classList.toggle("active");
    });

    // Close theme menu when clicking outside
    document.addEventListener("click", function (e) {
        if (themeButton && themeMenu && !e.target.closest(".theme-selector")) {
            themeMenu.classList.remove("show");
            themeButton.classList.remove("active");
        }
    });

    // Theme item click handlers
    themeItems.forEach(item => {
        item.addEventListener("click", function (e) {
            e.stopPropagation();
            const theme = this.getAttribute("data-theme");
            applyTheme(theme);

            // Close menu
            themeMenu.classList.remove("show");
            themeButton.classList.remove("active");

            // Visual feedback
            updateStatus(`✨ Theme switched to: ${theme.charAt(0).toUpperCase() + theme.slice(1)}`);
        });
    });
}

// Auto-resize textarea
messageInput.addEventListener("input", function () {
    this.style.height = "auto";
    this.style.height = Math.min(this.scrollHeight, 150) + "px";
});

// Custom Dropdown Functionality - Only if elements exist
if (dropdownButton && dropdownMenu && dropdownText) {
    dropdownButton.addEventListener("click", function (e) {
        e.stopPropagation();
        e.preventDefault();
        dropdownMenu.classList.toggle("show");
        dropdownButton.classList.toggle("active");
    });

    // Close dropdown when clicking outside
    document.addEventListener("click", function (e) {
        if (dropdownButton && dropdownMenu && !e.target.closest(".custom-dropdown")) {
            dropdownMenu.classList.remove("show");
            dropdownButton.classList.remove("active");
        }
    });

    // Handle dropdown item selection
    dropdownItems.forEach(item => {
        item.addEventListener("click", function (e) {
            e.stopPropagation();
            const value = this.getAttribute("data-value");

            // Update select element
            if (modeSelect) {
                modeSelect.value = value;
            }

            // Update dropdown button text
            if (dropdownText) {
                dropdownText.textContent = this.textContent.trim();
            }

            // Update selected item
            dropdownItems.forEach(i => i.classList.remove("selected"));
            this.classList.add("selected");

            // Close dropdown
            if (dropdownMenu && dropdownButton) {
                dropdownMenu.classList.remove("show");
                dropdownButton.classList.remove("active");
            }

            // Trigger mode change
            handleModeChange(value);

            // Update dropdown text with icon if needed
            const selectedItem = this.textContent.trim();
            if (dropdownText) {
                dropdownText.textContent = selectedItem.replace(/^[^\s]+\s/, ''); // Remove icon emoji
            }
        });
    });
}

// Handle mode change with animation
function handleModeChange(value) {
    const previousMode = currentMode;
    currentMode = value;

    // Add switching animation class
    dropdownButton.classList.add("switching");

    // Update status with mode name formatting
    const modeLabels = {
        "all_subjects": "All Subjects",
        "physics": "Physics",
        "mathematics": "Mathematics",
        "chemistry": "Chemistry",
        "astronomy": "Astronomy"
    };
    const modeName = modeLabels[value] || value.charAt(0).toUpperCase() + value.slice(1);
    updateStatus(`✨ Subject switched to: ${modeName}`);

    // Remove animation class after animation completes
    setTimeout(() => {
        dropdownButton.classList.remove("switching");
    }, 700);

    // Add visual feedback with a brief glow effect
    const navbar = document.querySelector('.navbar');
    if (navbar) {
        navbar.style.boxShadow =
            '0 4px 20px rgba(0, 0, 0, 0.3), 0 0 0 2px rgba(255, 102, 216, 0.2) inset, 0 0 60px rgba(255, 102, 216, 0.4)';

        setTimeout(() => {
            navbar.style.boxShadow = '';
        }, 800);
    }
}

// Sync dropdown with select changes (for compatibility)
if (modeSelect && dropdownText) {
    modeSelect.addEventListener("change", function () {
        handleModeChange(this.value);
        if (dropdownText) {
            const selectedText = this.options[this.selectedIndex].text;
            dropdownText.textContent = selectedText;
        }
        if (dropdownItems.length > 0) {
            dropdownItems.forEach(item => {
                item.classList.remove("selected");
                if (item.getAttribute("data-value") === this.value) {
                    item.classList.add("selected");
                }
            });
        }
    });
}

// Handle form submission
chatForm.addEventListener("submit", async function (e) {
    e.preventDefault();
    const message = messageInput.value.trim();
    if (!message || isProcessing) return;

    await sendMessage(message);
});

// Handle slash commands
function handleSlashCommand(message) {
    const trimmed = message.trim();

    if (trimmed.startsWith("/category ")) {
        const question = trimmed.substring("/category ".length).trim();
        if (!question) {
            return {
                isCommand: true,
                response: "Please provide a question after /category. Example: /category What is photosynthesis?"
            };
        }
        return {
            isCommand: true,
            command: "category",
            question: question,
            mode: "category_classifier"
        };
    }

    if (trimmed === "/help" || trimmed === "/help ") {
        return {
            isCommand: true,
            response: `**Available Commands:**

\`/category [question]\` - Find which category a question belongs to (Physics, Mathematics, Chemistry, or Astronomy)
  Example: /category What is Newton's second law?

\`/help\` - Show this help message

**Categories:**
- **All Subjects** - Answers questions from all categories
- **Physics** - Only Physics questions
- **Mathematics** - Only Mathematics questions  
- **Chemistry** - Only Chemistry questions
- **Astronomy** - Only Astronomy questions

Select a category from the dropdown to focus on a specific subject.`
        };
    }

    return { isCommand: false };
}

// Send message to backend
async function sendMessage(message) {
    isProcessing = true;
    updateUIForSending(message);

    try {
        // Check for slash commands
        const command = handleSlashCommand(message);

        if (command.isCommand && command.response) {
            // Handle built-in commands that return immediate responses
            chatHistory.push({ role: "user", text: message });
            addMessageToChat("user", message);
            chatHistory.push({ role: "model", text: command.response });
            addMessageToChat("model", command.response);
            updateStatus("Ready");

            messageInput.value = "";
            messageInput.style.height = "auto";
            isProcessing = false;
            sendButton.disabled = false;
            messageInput.focus();
            return;
        }

        if (command.isCommand && command.command === "category") {
            // Handle /category command - use category_classifier mode
            chatHistory.push({ role: "user", text: message });
            addMessageToChat("user", message);
            messageInput.value = "";
            messageInput.style.height = "auto";
            updateStatus("Classifying...");
            sendButton.disabled = true;

            const response = await fetch(`${API_BASE}/chat`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    message: command.question,
                    history: [],
                    mode: command.mode,
                }),
            });

            if (!response.ok) {
                let msg = "Failed to get response";
                try {
                    const error = await response.json();
                    msg = error.error || msg;
                } catch (_) {
                    try { msg = await response.text(); } catch (_) {}
                }
                throw new Error(`${response.status} ${response.statusText}: ${msg}`);
            }

            let data;
            try {
                data = await response.json();
            } catch (_) {
                const text = await response.text();
                throw new Error(text && text.trim().startsWith("<")
                    ? "Server returned HTML. Open the app via http://localhost:8000 while Flask is running."
                    : (text || "Invalid JSON from server"));
            }
            chatHistory.push({ role: "model", text: data.reply });
            addMessageToChat("model", data.reply);
            updateStatus("Ready");

            isProcessing = false;
            sendButton.disabled = false;
            messageInput.focus();
            return;
        }

        // Regular message handling
        // Add user message to history
        chatHistory.push({ role: "user", text: message });

        // Show user message in UI
        addMessageToChat("user", message);

        // Clear input
        messageInput.value = "";
        messageInput.style.height = "auto";

        updateStatus("Thinking...");
        sendButton.disabled = true;

        // Call API
        const response = await fetch(`${API_BASE}/chat`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                message: message,
                history: chatHistory,
                mode: currentMode,
            }),
        });

        if (!response.ok) {
            let msg = "Failed to get response";
            try {
                const error = await response.json();
                msg = error.error || msg;
            } catch (_) {
                try { msg = await response.text(); } catch (_) {}
            }
            throw new Error(`${response.status} ${response.statusText}: ${msg}`);
        }

        let data;
        try {
            data = await response.json();
        } catch (_) {
            const text = await response.text();
            throw new Error(text && text.trim().startsWith("<")
                ? "Server returned HTML. Open the app via http://localhost:8000 while Flask is running."
                : (text || "Invalid JSON from server"));
        }

        // Add assistant response to history
        chatHistory.push({ role: "model", text: data.reply });

        // Show assistant response in UI
        addMessageToChat("model", data.reply);

        updateStatus("Ready");
    } catch (error) {
        console.error("Error:", error);
        addMessageToChat("error", `Error: ${error.message}`);
        updateStatus("Error occurred");
    } finally {
        isProcessing = false;
        sendButton.disabled = false;
        messageInput.focus();
    }
}

// Update UI when sending
function updateUIForSending(message) {
    // Remove welcome message if present
    const welcomeMsg = chatMessages.querySelector(".welcome-message");
    if (welcomeMsg) {
        welcomeMsg.remove();
    }
}

// Add message to chat UI
function addMessageToChat(role, text) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${role}-message`;

    if (role === "user") {
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-text">${escapeHtml(text)}</div>
            </div>
        `;
    } else if (role === "error") {
        messageDiv.innerHTML = `
            <div class="message-content error-content">
                <div class="message-text">${escapeHtml(text)}</div>
            </div>
        `;
    } else {
        // Model response - format code blocks
        const formattedText = formatCodeBlocks(escapeHtml(text));
        messageDiv.innerHTML = `
            <div class="message-avatar">🤖</div>
            <div class="message-content">
                <div class="message-text">${formattedText}</div>
            </div>
        `;
    }

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Format code blocks with syntax highlighting
function formatCodeBlocks(text) {
    // Convert markdown code blocks to HTML
    text = text.replace(/```(\w+)?\n([\s\S]*?)```/g, function (match, lang, code) {
        const language = lang || "text";
        return `<pre class="code-block"><code class="language-${language}">${code.trim()}</code></pre>`;
    });

    // Convert inline code
    text = text.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');

    // Convert newlines to <br>
    text = text.replace(/\n/g, "<br>");

    return text;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// Update status text
function updateStatus(text) {
    statusText.textContent = text;
}

// Clear chat
clearButton.addEventListener("click", function () {
    chatHistory = [];
    chatMessages.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">📚</div>
            <h2>Welcome to Educational Assistant</h2>
            <p>Ask me about:</p>
            <ul>
                <li>Physics: Mechanics, Thermodynamics, Quantum Physics, Optics</li>
                <li>Mathematics: Algebra, Calculus, Geometry, Statistics</li>
                <li>Chemistry: Chemical Reactions, Organic Chemistry, Biochemistry</li>
                <li>Astronomy: Planets, Stars, Galaxies, Cosmology</li>
            </ul>
            <p class="example-query">Try: "What is Newton's second law of motion?"</p>
            <div class="slash-commands-info">
                <p class="commands-title">💡 Available Commands:</p>
                <div class="commands-list">
                    <div class="command-item">
                        <code>/category</code>
                        <span>Find which category a question belongs to</span>
                    </div>
                    <div class="command-item">
                        <code>/help</code>
                        <span>Show all available commands</span>
                    </div>
                </div>
                <p class="commands-example">Example: <code>/category What is photosynthesis?</code></p>
            </div>
        </div>
    `;
    updateStatus("Ready");
});

// Handle Enter key (submit) vs Shift+Enter (new line)
messageInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event("submit"));
    }
});

// Focus input on load
window.addEventListener("load", function () {
    messageInput.focus();
});

