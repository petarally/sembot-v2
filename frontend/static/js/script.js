document.addEventListener("DOMContentLoaded", function () {
  const chatForm = document.getElementById("chatForm");
  const userInput = document.getElementById("userInput");
  const chatMessages = document.getElementById("chatMessages");
  const suggestionButtons = document.querySelectorAll(".suggestion-btn");
  const suggestionsContainer = document.querySelector(".suggestions");

  // Backend API URL - now loaded from config.js
  const API_URL = CONFIG.API_URL;

  // Flag to track if first question was asked
  let firstQuestionAsked = false;

  // Slušač događaja za slanje poruke
  chatForm.addEventListener("submit", function (e) {
    e.preventDefault();
    sendMessage(userInput.value);
  });

  // Slušači događaja za gumbe s prijedlozima
  suggestionButtons.forEach((button) => {
    button.addEventListener("click", function () {
      sendMessage(this.textContent.trim());
    });
  });

  // Funkcija za slanje poruke
  function sendMessage(message) {
    if (!message.trim()) return;

    // Dodaj korisničku poruku u chat
    addMessage(message, "user");

    // Resetiraj input polje
    userInput.value = "";

    // Prikaži indikator učitavanja
    const loadingMessage = addMessage(
      "Učitavanje...",
      "bot",
      "loading-message"
    );

    // Pošalji zahtjev na API
    fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query: message }),
    })
      .then((response) => response.json())
      .then((data) => {
        // Ukloni indikator učitavanja
        chatMessages.removeChild(loadingMessage);

        // Dodaj odgovor bota
        addMessage(data.answer, "bot");

        // Update first question flag
        firstQuestionAsked = true;

        // Update suggested questions if available
        if (data.suggested_questions && data.suggested_questions.length > 0) {
          updateSuggestedQuestions(data.suggested_questions);
        }
      })
      .catch((error) => {
        // Ukloni indikator učitavanja
        chatMessages.removeChild(loadingMessage);

        // Prikaži poruku o grešci
        addMessage(
          "Žao mi je, došlo je do greške prilikom obrade vašeg upita. Molimo pokušajte ponovno.",
          "bot"
        );
        console.error("Error:", error);
      });
  }

  // Funkcija za ažuriranje prijedloga pitanja
  function updateSuggestedQuestions(questions) {
    // Promijeni naslov iz "Popularna pitanja" u "Predložena pitanja"
    const suggestionsTitle = suggestionsContainer.querySelector("h3");
    suggestionsTitle.textContent = "Predložena pitanja:";

    // Dohvati kontejner za gumbe
    const buttonContainer = suggestionsContainer.querySelector(
      ".suggestion-buttons"
    );

    // Očisti postojeće gumbe
    buttonContainer.innerHTML = "";

    // Dodaj nove gumbe za svako predloženo pitanje
    questions.forEach((question) => {
      const button = document.createElement("button");
      button.className = "suggestion-btn";
      button.textContent = question;

      // Dodaj event listener za novo pitanje
      button.addEventListener("click", function () {
        sendMessage(this.textContent.trim());
      });

      buttonContainer.appendChild(button);
    });
  }

  // Funkcija za dodavanje poruke u chat
  function addMessage(content, sender, className = "") {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${sender} ${className}`;

    const messageContent = document.createElement("div");
    messageContent.className = "message-content";

    // Format URLs, emails and phone numbers as links
    if (sender === "bot") {
      // Format links (http/https URLs)
      content = content.replace(
        /(https?:\/\/[^\s]+)/g,
        '<a href="$1" target="_blank" rel="noopener">$1</a>'
      );

      // Format email addresses
      content = content.replace(
        /([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9._-]+)/g,
        '<a href="mailto:$1">$1</a>'
      );

      // Format phone numbers (simple pattern)
      content = content.replace(
        /(\+\d{1,3}\s?)?(\(?\d{3,5}\)?[\s.-]?\d{3}[\s.-]?\d{3,4})/g,
        '<a href="tel:$1$2">$&</a>'
      );

      // Use innerHTML for bot messages with formatted links
      messageContent.innerHTML = content;
    } else {
      // Use textContent for user messages (safer)
      messageContent.textContent = content;
    }

    messageDiv.appendChild(messageContent);
    chatMessages.appendChild(messageDiv);

    // Scroll do najnovije poruke
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return messageDiv;
  }
});
