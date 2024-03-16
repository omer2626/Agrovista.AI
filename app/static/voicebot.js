const recognition = new webkitSpeechRecognition();
recognition.lang = 'en-US';
recognition.continuous = true;
recognition.interimResults = true;

recognition.onresult = function (event) {
    const last = event.results.length - 1;
    const command = event.results[last][0].transcript.trim().toLowerCase();

    switch (command) {
        case 'go to homepage':
            window.location.href = 'home';
            break;
        case 'go to login page':
            window.location.href = 'login';
            break;
        case 'go to service page':
            window.location.href = 'services';
            break;
        case 'scroll up':
            window.scrollBy(0, -100);
            break;
        case 'scroll down':
            window.scrollBy(0, 100);
            break;
        case 'scroll to top':
            window.scrollTo(0, 0);
            break;
        case 'scroll to bottom':
            window.scrollTo(0, document.body.scrollHeight);
            break;
        default:
            console.log('Command not recognized:', command);
    }
};

recognition.onend = function () {
    recognition.start(); // Restart recognition when it ends
};

// Start listening if it was active on the previous page
if (sessionStorage.getItem('micActive') === 'true') {
    recognition.start();
}

// Function to toggle the microphone
function toggleMic() {
    if (sessionStorage.getItem('micActive') === 'true') {
        recognition.stop();
        sessionStorage.setItem('micActive', 'false');
    } else {
        recognition.start();
        sessionStorage.setItem('micActive', 'true');
    }
}

// Start the microphone automatically when the page loads
window.onload = function () {
    toggleMic();
};
