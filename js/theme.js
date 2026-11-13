/* theme.js: light / dark mode.
   Loaded in <head> so the saved theme is applied before the page is drawn (no white flash).
   The click is handled here, so this file does not depend on app.js. */

const THEME_STORAGE_KEY = "fedxray-theme";

// Saved choice first, otherwise the computer's own light/dark setting
function getStartingTheme() {
  try {
    const savedTheme = localStorage.getItem(THEME_STORAGE_KEY);
    if (savedTheme === "dark" || savedTheme === "light") return savedTheme;
  } catch (error) {
    /* storage can be blocked by browser settings: ignore and fall through */
  }
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

// Make the button text match the current theme (the icons are switched by CSS)
function updateToggleLabel(theme) {
  const label = document.getElementById("theme-toggle-label");
  if (label) {
    label.textContent = theme === "dark" ? "Light mode" : "Dark mode";
  }
}

// Setting data-theme on <html> switches the colour tokens in base.css
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  updateToggleLabel(theme);
}

function toggleTheme() {
  const newTheme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  applyTheme(newTheme);
  try {
    localStorage.setItem(THEME_STORAGE_KEY, newTheme);
  } catch (error) {
    /* not saved, but the theme still changes for this visit */
  }
}

// One click listener for the whole page: it works whenever the button exists
document.addEventListener("click", event => {
  if (event.target.closest("#theme-toggle")) toggleTheme();
});

// app.js calls this. It only fixes the label, because the click is handled above.
function initThemeToggle() {
  updateToggleLabel(document.documentElement.dataset.theme);
}

applyTheme(getStartingTheme());                        // runs immediately
document.addEventListener("DOMContentLoaded", initThemeToggle);