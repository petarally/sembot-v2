// Mini admin sučelje za punjenje baze pitanja bez redeploya.
const ADMIN_API = CONFIG.API_BASE_URL + "/admin/qa";

const tokenInput = document.getElementById("token");
const statusBox = document.getElementById("status");
const listBox = document.getElementById("list");
const countEl = document.getElementById("count");

// Token se pamti lokalno (zgodno, ali nije visoka sigurnost — vidi README).
tokenInput.value = localStorage.getItem("adminToken") || "";
tokenInput.addEventListener("change", () =>
  localStorage.setItem("adminToken", tokenInput.value)
);

function headers() {
  return { "Content-Type": "application/json", "X-Admin-Token": tokenInput.value };
}

function showStatus(message, ok) {
  statusBox.textContent = message;
  statusBox.className = ok ? "ok" : "err";
}

async function loadList() {
  try {
    const res = await fetch(ADMIN_API, { headers: headers() });
    if (!res.ok) throw new Error((await res.json()).detail || res.status);
    const items = await res.json();
    countEl.textContent = items.length;
    listBox.innerHTML = "";
    items.forEach((qa) => {
      const row = document.createElement("div");
      row.className = "qa";
      const text = document.createElement("span");
      text.innerHTML = `<strong>${escapeHtml(qa.question)}</strong><br><small>${escapeHtml(qa.answer)}</small>`;
      const btn = document.createElement("button");
      btn.className = "del";
      btn.textContent = "Obriši";
      btn.onclick = () => remove(qa.id, qa.question);
      row.append(text, btn);
      listBox.appendChild(row);
    });
  } catch (e) {
    showStatus("Greška pri dohvaćanju: " + e.message, false);
  }
}

async function add() {
  const question = document.getElementById("question").value.trim();
  const answer = document.getElementById("answer").value.trim();
  const paraphrases = document
    .getElementById("paraphrases")
    .value.split("\n")
    .map((s) => s.trim())
    .filter(Boolean);

  if (!question || !answer) {
    showStatus("Pitanje i odgovor su obavezni.", false);
    return;
  }
  try {
    const res = await fetch(ADMIN_API, {
      method: "POST",
      headers: headers(),
      body: JSON.stringify({ question, answer, paraphrases }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.status);
    document.getElementById("question").value = "";
    document.getElementById("answer").value = "";
    document.getElementById("paraphrases").value = "";
    showStatus("Dodano.", true);
    loadList();
  } catch (e) {
    showStatus("Greška pri dodavanju: " + e.message, false);
  }
}

async function remove(id, question) {
  if (!confirm(`Obrisati pitanje?\n\n${question}`)) return;
  try {
    const res = await fetch(`${ADMIN_API}/${id}`, { method: "DELETE", headers: headers() });
    if (!res.ok) throw new Error((await res.json()).detail || res.status);
    showStatus("Obrisano.", true);
    loadList();
  } catch (e) {
    showStatus("Greška pri brisanju: " + e.message, false);
  }
}

async function bulkImport() {
  const file = document.getElementById("bulkFile").files[0];
  const pasted = document.getElementById("bulkText").value.trim();

  let raw;
  try {
    raw = file ? await file.text() : pasted;
    if (!raw) {
      showStatus("Odaberi datoteku ili zalijepi JSON.", false);
      return;
    }
    const parsed = JSON.parse(raw);
    // Prihvati i goli niz i {"qa_pairs":[...]} (kao qa_data.json).
    const items = Array.isArray(parsed) ? parsed : parsed.qa_pairs;
    if (!Array.isArray(items) || items.length === 0) {
      throw new Error("očekujem neprazan niz pitanja");
    }
    const res = await fetch(`${ADMIN_API}/bulk`, {
      method: "POST",
      headers: headers(),
      body: JSON.stringify({ items }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.status);
    document.getElementById("bulkFile").value = "";
    document.getElementById("bulkText").value = "";
    showStatus(`Uvezeno ${data.added} pitanja.`, true);
    loadList();
  } catch (e) {
    showStatus("Greška pri uvozu: " + e.message, false);
  }
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

document.getElementById("addBtn").addEventListener("click", add);
document.getElementById("bulkBtn").addEventListener("click", bulkImport);
loadList();
