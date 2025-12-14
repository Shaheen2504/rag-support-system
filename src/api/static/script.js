document.getElementById("chat-form").addEventListener("submit", async (e) => {
    e.preventDefault();

    const questionInput = document.getElementById("question");
    const responseContainer = document.getElementById("response");

    const question = questionInput.value.trim();
    if (!question) return;

    // Clear input
    questionInput.value = "";

    // Add user's question to the response container
    const userMessage = document.createElement("p");
    userMessage.textContent = `You: ${question}`;
    userMessage.style.color = "#1976d2"; // Blue for user messages
    responseContainer.appendChild(userMessage);

    // Add a loading spinner
    const loadingMessage = document.createElement("p");
    loadingMessage.textContent = "Bot: Thinking...";
    loadingMessage.style.color = "#888";
    responseContainer.appendChild(loadingMessage);

    try {
        // Send question to the FastAPI backend
        const response = await fetch("/answer", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ question }),
