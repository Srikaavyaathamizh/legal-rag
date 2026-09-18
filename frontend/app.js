const API_URL = "http://127.0.0.1:8000/query";

const questionInput = document.getElementById("question");
const incidentDate = document.getElementById("incidentDate");
const sendButton = document.getElementById("sendButton");

const welcome = document.getElementById("welcome");
const chat = document.getElementById("chat");
const messages = document.getElementById("messages");


// ============================================================
// SUGGESTION
// ============================================================

function useSuggestion(text) {

    questionInput.value = text;

    questionInput.focus();

}


// ============================================================
// ENTER KEY
// ============================================================

function handleKeyDown(event) {

    if (
        event.key === "Enter" &&
        !event.shiftKey
    ) {

        event.preventDefault();

        sendQuestion();

    }

}


// ============================================================
// SEND QUESTION
// ============================================================

async function sendQuestion() {

    const question = questionInput.value.trim();

    if (!question) {

        questionInput.focus();

        return;

    }


    // --------------------------------------------------------
    // Switch from welcome screen to chat
    // --------------------------------------------------------

    welcome.classList.add("hidden");

    chat.classList.remove("hidden");


    // --------------------------------------------------------
    // Add user message
    // --------------------------------------------------------

    addUserMessage(question);


    // --------------------------------------------------------
    // Clear input
    // --------------------------------------------------------

    questionInput.value = "";


    // --------------------------------------------------------
    // Loading
    // --------------------------------------------------------

    const loadingId = addLoadingMessage();


    sendButton.disabled = true;


    try {

        const response = await fetch(API_URL, {

            method: "POST",

            headers: {

                "Content-Type": "application/json"

            },

            body: JSON.stringify({

                question: question,

                incident_date:
                    incidentDate.value || null

            })

        });


        const data = await response.json();


        removeMessage(loadingId);


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "The server returned an error."
            );

        }


        addAssistantMessage(data);


    }

    catch (error) {

        removeMessage(loadingId);

        addErrorMessage(error.message);

    }

    finally {

        sendButton.disabled = false;

        questionInput.focus();

    }

}


// ============================================================
// USER MESSAGE
// ============================================================

function addUserMessage(text) {

    const message = document.createElement("div");

    message.className = "message user";


    message.innerHTML = `

        <div class="message-avatar">
            👤
        </div>

        <div class="message-content">
            ${escapeHtml(text)}
        </div>

    `;


    messages.appendChild(message);

    scrollToBottom();

}


// ============================================================
// ASSISTANT MESSAGE
// ============================================================

function addAssistantMessage(data) {

    const llm = data.llm_result || {};

    const responseText =
        llm.response_text ||
        data.answer ||
        "No answer was returned.";


    const law =
        data.applicable_law ||
        "Unknown";


    const trustworthy =
        llm.trustworthy;


    const incident =
        data.incident_date ||
        "";


    let html = `

        <div class="answer-title">
            Legal RAG Answer
        </div>

        <div class="answer-text">
            ${formatAnswer(responseText)}
        </div>

        <div class="meta">

            <span class="badge law">
                ⚖ ${escapeHtml(law)}
            </span>

    `;


    if (incident) {

        html += `

            <span class="badge">
                📅 ${escapeHtml(incident)}
            </span>

        `;

    }


    if (trustworthy === true) {

        html += `

            <span class="badge good">
                ✓ Grounded
            </span>

        `;

    }


    html += `</div>`;


    // --------------------------------------------------------
    // Sources
    // --------------------------------------------------------

    const sections =
        data.retrieved_sections ||
        data.evidence_sections ||
        [];


    if (sections.length > 0) {

        html += `

            <div class="sources">

                <div class="sources-title">
                    Retrieved Sources
                </div>

        `;


        sections
            .slice(0, 5)
            .forEach(section => {

                const lawName =
                    section.law || "";

                const sectionNumber =
                    section.section || "";

                const heading =
                    section.heading ||
                    "Legal provision";

                const text =
                    section.full_text ||
                    section.text ||
                    "";


                html += `

                    <div class="source-card">

                        <div class="source-heading">
                            ${escapeHtml(lawName)}
                            Section
                            ${escapeHtml(
                                String(sectionNumber)
                            )}
                            — ${escapeHtml(heading)}
                        </div>

                        <div class="source-text">
                            ${escapeHtml(
                                truncate(text, 350)
                            )}
                        </div>

                    </div>

                `;

            });


        html += `</div>`;

    }


    const message = document.createElement("div");

    message.className = "message assistant";


    message.innerHTML = `

        <div class="message-avatar">
            ⚖
        </div>

        <div class="message-content">

            ${html}

        </div>

    `;


    messages.appendChild(message);

    scrollToBottom();

}


// ============================================================
// ERROR
// ============================================================

function addErrorMessage(error) {

    const message = document.createElement("div");

    message.className = "message assistant";


    message.innerHTML = `

        <div class="message-avatar">
            ⚠
        </div>

        <div class="message-content">

            <div class="answer-title">
                Unable to process the question
            </div>

            <div class="answer-text">
                ${escapeHtml(error)}
            </div>

        </div>

    `;


    messages.appendChild(message);

    scrollToBottom();

}


// ============================================================
// LOADING
// ============================================================

function addLoadingMessage() {

    const id =
        "loading-" +
        Date.now();


    const message =
        document.createElement("div");


    message.id = id;

    message.className =
        "message assistant";


    message.innerHTML = `

        <div class="message-avatar">
            ⚖
        </div>

        <div class="message-content">

            <div class="loading">

                <span></span>
                <span></span>
                <span></span>

                <span style="
                    width:auto;
                    height:auto;
                    background:none;
                    margin-left:5px;
                ">
                    Searching legal sources...
                </span>

            </div>

        </div>

    `;


    messages.appendChild(message);

    scrollToBottom();


    return id;

}


// ============================================================
// REMOVE MESSAGE
// ============================================================

function removeMessage(id) {

    const element =
        document.getElementById(id);


    if (element) {

        element.remove();

    }

}


// ============================================================
// FORMAT ANSWER
// ============================================================

function formatAnswer(text) {

    let result =
        escapeHtml(String(text));


    // Bold markdown
    result =
        result.replace(
            /\*\*(.*?)\*\*/g,
            "<strong>$1</strong>"
        );


    // New lines
    result =
        result.replace(
            /\n/g,
            "<br>"
        );


    return result;

}


// ============================================================
// ESCAPE HTML
// ============================================================

function escapeHtml(value) {

    return String(value)

        .replace(/&/g, "&amp;")

        .replace(/</g, "&lt;")

        .replace(/>/g, "&gt;")

        .replace(/"/g, "&quot;")

        .replace(/'/g, "&#039;");

}


// ============================================================
// TRUNCATE
// ============================================================

function truncate(text, length) {

    text = String(text || "");


    if (text.length <= length) {

        return text;

    }


    return text.substring(0, length) + "...";

}


// ============================================================
// SCROLL
// ============================================================

function scrollToBottom() {

    setTimeout(() => {

        const main =
            document.querySelector(".main");


        main.scrollTo({

            top: main.scrollHeight,

            behavior: "smooth"

        });

    }, 50);

}