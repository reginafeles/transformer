document.addEventListener('DOMContentLoaded', () => {
      const btn = document.getElementById('submit-btn');
      const input = document.getElementById('text-input');
      const resultBox = document.getElementById('result-box');

      async function sendFeedback(isCorrect) {
        const text = input.value.trim();
        const predictedLabel = resultBox.textContent.split(":")[1].split(",")[0].trim();
        await fetch('/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text, predicted_label: predictedLabel, correct: isCorrect })
        });
      }

      btn.addEventListener('click', async () => {
        const text = input.value.trim();
        if (!text) {
          resultBox.textContent = "Введите текст";
          resultBox.className = "alert alert-warning mt-4";
          return;
        }

        resultBox.textContent = "⏳ Анализ...";
        resultBox.className = "alert alert-info mt-4";

        try {
          const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
          });

          const data = await response.json();
          resultBox.textContent = `🏷 Класс: ${data.label}, уверенность: ${data.confidence}`;
          resultBox.className = "alert alert-success mt-4";

          document.getElementById("feedback-section").classList.remove("d-none");
          document.getElementById("probs-container").style.display = 'block';
          const probsBars = document.getElementById('probs-bars');
          probsBars.innerHTML = '';
          for (const [label, prob] of Object.entries(data.probs)) {
            const bar = document.createElement('div');
            bar.className = 'progress my-2';

            const inner = document.createElement('div');
          inner.className = 'progress-bar';
          inner.style.width = `${prob * 100}%`;
          inner.style.color = 'black';
           inner.innerText = `${label}: ${(prob * 100).toFixed(1)}%`;


            bar.appendChild(inner);
            probsBars.appendChild(bar);
          }
        } catch (err) {
          resultBox.textContent = "Ошибка при анализе текста";
          resultBox.className = "alert alert-danger mt-4";
        }
      });

      document.getElementById("btn-correct").addEventListener("click", async () => {
        await sendFeedback(true);
        document.getElementById("feedback-section").classList.add("d-none");
        resultBox.textContent += "<\br>✅ Спасибо за подтверждение!";
      });

      document.getElementById("btn-incorrect").addEventListener("click", () => {
        sendFeedback(false);
        document.getElementById("correction-form").classList.remove("d-none");
      });

      document.getElementById("send-correction").addEventListener("click", () => {
        const correctLabel = document.getElementById("correct-label").value;
        const text = document.getElementById("text-input").value.trim();
        const predictedLabel = resultBox.textContent.split(":")[1].split(",")[0].trim();

        if (!correctLabel) {
          alert("Выберите правильный класс");
          return;
        }

        fetch('/correction', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text, predicted_label: predictedLabel, true_label: correctLabel })
        }).then(() => {
          alert("Спасибо, исправление отправлено");
          document.getElementById("correction-form").classList.add("d-none");
        });
      });

      const themeToggle = document.getElementById("theme-toggle");
      themeToggle.addEventListener("change", () => {
        document.body.classList.toggle("bg-dark");
        document.body.classList.toggle("text-white");
        document.querySelector(".container").classList.toggle("bg-dark");
        document.querySelector(".container").classList.toggle("text-white");
        document.body.classList.toggle("dark-theme");
        document.querySelectorAll('h1, .form-label').forEach(el => {
        el.classList.toggle('text-dark');
    })
      });
    });