"""
scraper.py — Web Scraper for IIT Jodhpur Corpus
=================================================
This module fetches textual content from IIT Jodhpur web pages.

The scraper uses requests + BeautifulSoup to pull HTML and extract
only the meaningful body text, discarding navigation, scripts, and
other boilerplate elements.
"""

import os
import re
import time
import requests
from bs4 import BeautifulSoup

REQUEST_DELAY = 1.5  # seconds between consecutive fetches
REQUEST_TIMEOUT = 20  # seconds before giving up on a page
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ------------------------------------------------------------------
# Source URLs grouped by category
# Each tuple: (readable_label, url)
# We keep at least three distinct source categories.
#------------------------------------------------------------------

# ── Source 1: Official Website Pages ──────────────────────────────
OFFICIAL_PAGES = [
    ("introduction", "https://iitj.ac.in/main/en/introduction"),
    ("about_iitj", "https://iitj.ac.in/main/en/iitj"),
    ("director_message", "https://iitj.ac.in/main/en/director"),
    ("chairman_message", "https://iitj.ac.in/main/en/chairman"),
    ("why_career_iitj", "https://iitj.ac.in/main/en/why-pursue-a-career-@-iit-jodhpur"),
    ("campus_life", "https://iitj.ac.in/office-of-students/en/campus-life"),
    (
        "office_of_students",
        "https://iitj.ac.in/office-of-students/en/office-of-students",
    ),
    ("news_announcements", "https://iitj.ac.in/main/en/news"),
    ("events", "https://iitj.ac.in/main/en/events"),
    ("annual_report", "https://iitj.ac.in/Main/en/Annual-Reports-of-the-Institute"),
]

# ── Source 2: Academic Regulations, Programs, Curriculum ─

ACADEMIC_PAGES = [
    # --- Academic Regulations ---
    (
        "academic_regulations",
        "https://iitj.ac.in/office-of-academics/en/academic-regulations",
    ),
    # --- Circulars ---
    ("academic_circulars", "https://iitj.ac.in/office-of-academics/en/circulars"),
    # --- Program Curriculum / Course Syllabus ---
    ("program_curriculum", "https://iitj.ac.in/office-of-academics/en/curriculum"),
    (
        "program_structure",
        "https://iitj.ac.in/office-of-academics/en/program-Structure",
    ),
    (
        "academic_programs_list",
        "https://iitj.ac.in/office-of-academics/en/list-of-academic-programs",
    ),
    # --- Individual program pages ---
    ("btech_program", "https://iitj.ac.in/office-of-academics/en/b.tech."),
    ("mtech_program", "https://iitj.ac.in/office-of-academics/en/m.tech."),
    ("phd_program", "https://iitj.ac.in/office-of-academics/en/ph.d."),
    ("mba_program", "https://iitj.ac.in/office-of-academics/en/mba"),
    ("itep_program", "https://iitj.ac.in/itep/"),
    # --- SAIDE course/syllabus pages ---
    (
        "saide_btech",
        "https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/btech",
    ),
    (
        "saide_mtech",
        "https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/mtech",
    ),
    (
        "saide_courses",
        "https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/courses",
    ),
    # --- Other academic info ---
    ("academics_office", "https://iitj.ac.in/office-of-academics/en/academics"),
    (
        "academic_calendar",
        "https://iitj.ac.in/Office-of-Academics/en/Academic-Calendar",
    ),
    (
        "ug_registration",
        "https://iitj.ac.in/office-of-academics/en/ug-registration-guidelines",
    ),
    (
        "pg_admissions",
        "https://iitj.ac.in/admission-postgraduate-programs/en/Admission-to-Postgraduate-Programs",
    ),
    (
        "executive_education",
        "https://iitj.ac.in/office-of-executive-education/en/office-of-executive-education",
    ),
    ("scholarships", "https://iitj.ac.in/office-of-academics/en/scholarships"),
    ("convocation", "https://iitj.ac.in/office-of-academics/en/convocation"),
    ("faqs_applicants", "https://iitj.ac.in/main/en/faqs-applicants"),
    (
        "office_registrar",
        "https://iitj.ac.in/office-of-registrar/en/office-of-registrar",
    ),
    (
        "office_administration",
        "https://iitj.ac.in/office-of-administration/en/office-of-administration",
    ),
]

# ── Source 3: Newsletters / Institute Repository ─────────────────
NEWSLETTER_PAGES = [
    ("newsletter", "https://iitj.ac.in/institute-repository/en/Newsletter"),
]

# ── Source 4: Department / School pages ──────────────────────────
DEPARTMENT_PAGES = [
    (
        "school_ai_ds",
        "https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/school-of-artificial-intelligence-and-data-science",
    ),
    ("engineering_science", "https://iitj.ac.in/es/en/engineering-science"),
    ("school_liberal_arts", "https://iitj.ac.in/school-of-liberal-arts/"),
    ("school_design", "https://iitj.ac.in/school-of-design/"),
    ("school_management", "https://iitj.ac.in/schools/"),
    ("departments_listing", "https://iitj.ac.in/m/Index/main-departments?lg=en"),
    ("centres_listing", "https://iitj.ac.in/m/Index/main-centers?lg=en"),
    ("idrps_idrcs", "https://iitj.ac.in/m/Index/main-idrps-idrcs?lg=en"),
    (
        "research_development",
        "https://iitj.ac.in/office-of-research-development/en/office-of-research-and-development",
    ),
    ("research_highlights", "https://iitj.ac.in/main/en/research-highlight"),
    ("central_research_facility", "https://iitj.ac.in/crf/en/crf"),
    ("techscape", "https://iitj.ac.in/techscape/en/Techscape"),
    ("health_center", "https://iitj.ac.in/health-center/en/health-center"),
]
# Merge everything into one flat list for convenience
ALL_SOURCES = (
    OFFICIAL_PAGES
    + ACADEMIC_PAGES
    + NEWSLETTER_PAGES
    + DEPARTMENT_PAGES
)


def fetch_page(url):
    """
    Download a single web page and return its raw HTML string.
    Returns None when the request fails so the caller can skip it.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        # only keep pages that are actually HTML
        content_type = response.headers.get("Content-Type", "")
        if "html" not in content_type.lower() and "text" not in content_type.lower():
            print(f"  [skip] Non-HTML content at {url}")
            return None
        return response.text
    except requests.RequestException as err:
        print(f"  [error] Could not fetch {url}: {err}")
        return None


def strip_boilerplate(html_text):
    """
    Parse raw HTML and return only the useful body text.

    Strategy:
      1. Build a BeautifulSoup tree.
      2. Remove all <script>, <style>, <nav>, <footer>, <header>,
         <noscript>, and <aside> elements — these are navigation,
         styling, or JS code that we never want in the corpus.
      3. Also remove any element whose class/id hints at menus,
         cookie banners, or sidebars.
      4. Extract the remaining visible text with get_text().
    """
    soup = BeautifulSoup(html_text, "html.parser")

    # Tags that never carry useful article text
    unwanted_tags = [
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "noscript",
        "aside",
        "iframe",
        "svg",
        "form",
        "button",
        "input",
        "select",
    ]
    for tag_name in unwanted_tags:
        for element in soup.find_all(tag_name):
            element.decompose()  # remove the element entirely

    # Class / id patterns that commonly wrap menus, banners, etc.
    boilerplate_hints = [
        "navbar",
        "nav-bar",
        "menu",
        "sidebar",
        "cookie",
        "footer",
        "header",
        "breadcrumb",
        "banner",
        "advertisement",
        "social-share",
        "share-buttons",
        "top-bar",
        "bottom-bar",
        "skip-link",
    ]
    for element in soup.find_all(True):
        # gather all class names and the id into one string to check
        attrs = element.attrs or {}
        classes = " ".join(attrs.get("class", []))
        elem_id = attrs.get("id", "") or ""
        combined = (classes + " " + elem_id).lower()
        if any(hint in combined for hint in boilerplate_hints):
            element.decompose()

    # Pull plain text out, collapsing whitespace
    text = soup.get_text(separator=" ", strip=True)

    # The new IITJ website injects token strings like
    #   ###147852369$$$_RedirectToLoginPage_%%%963258741!!!
    # and stray accessibility toggles ("A+ A A-").  Remove those.
    text = re.sub(r"###.*?!!!", " ", text)
    text = re.sub(r"A\+\s*A\s+A-", " ", text)

    # Remove any Hindi / Devanagari text (we only keep English)
    text = re.sub(r"[\u0900-\u097F]+", " ", text)

    # Collapse whitespace one more time
    text = re.sub(r"\s+", " ", text).strip()

    return text


def save_raw_page(label, text, output_dir):
    """
    Persist the extracted raw text for a page into the raw_pages folder.
    This intermediate file lets us inspect what was scraped before
    any further cleaning happens.
    """
    filepath = os.path.join(output_dir, f"{label}.txt")
    with open(filepath, "w", encoding="utf-8") as fout:
        fout.write(text)
    return filepath


def scrape_all_sources(output_dir):
    """
    Main entry point: iterate over every URL, download, strip
    boilerplate, and save raw text files.

    Returns
    -------
    collected : list[tuple[str, str]]
        Each element is (label, extracted_text) for pages that
        were fetched successfully.
    """
    os.makedirs(output_dir, exist_ok=True)

    collected = []
    total = len(ALL_SOURCES)

    for idx, (label, url) in enumerate(ALL_SOURCES, start=1):
        print(f"[{idx}/{total}] Fetching: {label}  —  {url}")
        html = fetch_page(url)
        if html is None:
            continue

        # strip away navigation, scripts, footers, etc.
        body_text = strip_boilerplate(html)

        # skip pages that yielded almost no text (< 50 chars)
        if len(body_text.strip()) < 50:
            print(f"  [skip] Too little text extracted for {label}")
            continue

        save_raw_page(label, body_text, output_dir)
        collected.append((label, body_text))
        print(f"  -> saved {len(body_text)} chars")

        # be polite: pause before the next request
        if idx < total:
            time.sleep(REQUEST_DELAY)

    print(f"\nScraping complete. {len(collected)}/{total} pages collected.")
    return collected
