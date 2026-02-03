async function sendMessage() {
  const input = document.getElementById("userInput");
  const msg = input.value;
  if (!msg) return;

  const messages = document.getElementById("messages");
  messages.innerHTML += `<p><b>You:</b> ${msg}</p>`;
  input.value = "";

  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: msg })
  });

  const data = await res.json();
  messages.innerHTML += `<p><b>Bot:</b> ${data.reply}</p>`;
}
