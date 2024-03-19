const recognition = new webkitSpeechRecognition();
recognition.lang = 'en-US';
recognition.continuous = true;
recognition.interimResults = true;

function isLoginPage() {
    return window.location.href.includes('login');
}


// recognition.onresult = function (event) {
//     const last = event.results.length - 1;
//     const command = event.results[last][0].transcript.trim().toLowerCase();
recognition.onresult = function (event) {
    const last = event.results.length - 1;
    const transcript = event.results[last][0].transcript.trim();
    const command = transcript.toLowerCase();

    // if (isLoginPage()) {
    //     console.log('listening login details....')
    //     if (command.startsWith('enter the phone number as')) {
    //         const phoneNumber = transcript.split(' ').slice(-1)[0];
    //         console.log("Extracted phone number:", phoneNumber);
    //         document.getElementById('phone_No').value = phoneNumber;
    //     } else if (command.startsWith('enter the password')) {
    //         const password = transcript.split(' ').slice(-1)[0];
    //         document.getElementById('password').value = password;
    //     }
    if (isLoginPage()) {
        console.log('listening login details....')
        const phoneNumberMatch = transcript.match(/\d+/g);
        if (phoneNumberMatch && command.includes('enter the phone number as')) {
            const phoneNumber = phoneNumberMatch.join('');
            console.log("Extracted phone number:", phoneNumber);
            document.getElementById('phone_No').value = phoneNumber;
        } else if (command.startsWith('enter the password')) {
            console.log("listening password details")
            const password = transcript.substring(transcript.indexOf('password') + 9).trim();
            console.log(password)
            document.getElementById('password').value = password;
        }

    } else {
        switch (command) {
            case 'go to homepage':
                window.location.href = 'home';
                break;
            case 'homepage':
                window.location.href = 'home';
                break;
            case 'go to login page':
                window.location.href = 'login';
                break;
            case 'login page':
                window.location.href = 'login';
                break;
            case 'go to service page':
                window.location.href = 'services';
                break;
            case 'service page':
                window.location.href = 'services';
                break;
            case 'go to weather page':
                window.location.href = 'weather';
                break;
            case 'weather page':
                window.location.href = 'weather';
                break;
            case 'go to pest detection page':
                window.location.href = 'pest_detection';
                break;
            case 'pest detection page':
                window.location.href = 'pest_detection';
                break;
            case 'test detection page':
                window.location.href = 'pest_detection';
                break;
            case 'best detection page':
                window.location.href = 'pest_detection';
                break;
            case 'go to fertilizer prediction page':
                window.location.href = 'fertilizer_recommendation';
                break;
            case 'fertilizer prediction page':
                window.location.href = 'fertilizer_recommendation';
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
