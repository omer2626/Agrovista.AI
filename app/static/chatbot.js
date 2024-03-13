function sendMessage() {
    var userInput = document.getElementById("userInput").value.toLowerCase().trim(); // Convert to lowercase and trim spaces
    var botMessages = document.getElementById("botMessages");

    var userMessage = document.createElement("div");
    userMessage.textContent = userInput;
    userMessage.className = "message user-message";
    botMessages.appendChild(userMessage);

    setTimeout(function() {
        var botResponse = document.createElement("div");
        var response = questionsAndAnswers[userInput] || "I'm sorry, I don't have an answer to that question.";
        botResponse.textContent = response;
        botResponse.className = "message bot-message";
        botMessages.appendChild(botResponse);

        botMessages.scrollTop = botMessages.scrollHeight;
    }, 1000);

    document.getElementById("userInput").value = ""; // Clear input field
}

function handleKeyPress(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
}
