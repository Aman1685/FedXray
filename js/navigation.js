/* navigation.js: builds the sidebar, switches between views and handles the mobile menu.
   Each sidebar link points to a hash such as #dataset. When the hash changes,
   showView() displays the matching <section id="view-dataset">. */

const sidebarNav = document.getElementById("sidebar-nav");
const pageTitle = document.getElementById("page-title");
const pageSubtitle = document.getElementById("page-subtitle");
const pendingView = document.getElementById("view-pending");
const pendingTitle = document.getElementById("pending-title");
const menuToggle = document.getElementById("menu-toggle");
const sidebarOverlay = document.getElementById("sidebar-overlay");

// One link (icon + title) per entry in SECTIONS (config.js)
function buildSidebar() {
  sidebarNav.innerHTML = SECTIONS.map(section => `
    <a class="nav-link" href="#${section.id}" data-view="${section.id}">
      ${iconSvg(section.icon)}<span>${section.title}</span>
    </a>`).join("");
}

// Show one view and mark its sidebar link as active
function showView(sectionId) {
  const section = SECTIONS.find(item => item.id === sectionId) || SECTIONS[0];

  // If the section has no HTML yet, fall back to the "not built yet" view
  const activeView = document.getElementById("view-" + section.id) || pendingView;

  document.querySelectorAll(".view").forEach(view => view.classList.remove("is-active"));
  activeView.classList.add("is-active");

  pendingTitle.textContent = section.title;
  pageTitle.textContent = section.title;
  pageSubtitle.textContent = section.description;
  document.title = section.title + " | FedXray";

  sidebarNav.querySelectorAll(".nav-link").forEach(link => {
    const isCurrent = link.dataset.view === section.id;
    link.classList.toggle("is-active", isCurrent);
    if (isCurrent) {
      link.setAttribute("aria-current", "page");
    } else {
      link.removeAttribute("aria-current");
    }
  });

  setSidebarOpen(false);
  window.scrollTo(0, 0);
}

// Mobile menu open/close
function setSidebarOpen(isOpen) {
  document.body.classList.toggle("sidebar-open", isOpen);
  menuToggle.setAttribute("aria-expanded", String(isOpen));
}

function handleHashChange() {
  showView(window.location.hash.slice(1)); // "#dataset" becomes "dataset"
}

buildSidebar();
window.addEventListener("hashchange", handleHashChange);
menuToggle.addEventListener("click", () => setSidebarOpen(!document.body.classList.contains("sidebar-open")));
sidebarOverlay.addEventListener("click", () => setSidebarOpen(false));
document.addEventListener("keydown", event => {
  if (event.key === "Escape") setSidebarOpen(false);
});
handleHashChange(); // show the correct view on first load