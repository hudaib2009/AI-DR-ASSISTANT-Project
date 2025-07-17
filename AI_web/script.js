// Utility Functions
function addClass(el, className) {
    if (el && !el.classList.contains(className)) {
        el.classList.add(className);
    }
}

function removeClass(el, className) {
    if (el && el.classList.contains(className)) {
        el.classList.remove(className);
    }
}

// Language Translations
const translations = {
    "en": {
        "app_title": "AI Lung cancer diagnosis",
        "change_lang": "Change Language",
        "select_xray": "Select X-Ray Image",
        "upload_xray": "Upload X-Ray",
        "select_ct": "Select CT Scan Image",
        "upload_ct": "Upload CT Scan",
        "result": "Result",
        "true": "Positive",
        "false": "Negative",
        "about": "About",
        "about_title": "About the Tool",
        "about_content": "This tool is designed to assist in medical diagnosis. It provides functionalities for analyzing X-ray and CT scan images, to find traces of cancerous cysts in the chest possibly related to lung cancer.  Please use it responsibly.",
        "general_idea": "How it Works",
        "idea_title": "General Idea",
        "idea_content": "The tool uses image processing techniques to analyze medical images and provide potential diagnostic insights. The results are for informational purposes only and should not replace a professional medical opinion.",
        "cancer_signs": "Cancer Symptoms",
        "cancer_title": "Cancer Symptoms",
        "cancer_content": "Common symptoms of cancer include: fatigue, unexplained weight loss, changes in bowel habits, persistent cough, and unusual bleeding or discharge. Please consult a doctor if you notice any of these.",
        "contact": "Contact",
        "contact_title": "Contact Information",
        "contact_content": "For inquiries or feedback, please contact us at: support@example.com",
        "available_langs": ["English", "Arabic", "French", "Spanish"],
        "close": "Close",
        "home": "Home"
    },
    "ar": {
        "app_title": "أداة التشخيص الطبي",
        "change_lang": "تغيير اللغة",
        "select_xray": "اختر صورة الأشعة السينية",
        "upload_xray": "تحميل الأشعة السينية",
        "select_ct": "اختر صورة الأشعة المقطعية",
        "upload_ct": "تحميل الأشعة المقطعية",
        "result": "النتيجة",
        "true": "إيجابي",
        "false": "سلبي",
        "about": "حول",
        "about_title": "حول الأداة",
        "about_content": "تم تصميم هذه الأداة للمساعدة في التشخيص الطبي. توفر وظائف لتحليل صور الأشعة السينية والأشعة المقطعية، للعثور على آثار الأكياس السرطانية في الصدر التي قد تكون مرتبطة بسرطان الرئة. يرجى استخدامها بمسؤولية.",
        "general_idea": "كيف تعمل",
        "idea_title": "الفكرة العامة",
        "idea_content": "تستخدم الأداة تقنيات معالجة الصور لتحليل الصور الطبية وتقديم رؤى تشخيصية محتملة. النتائج هي لأغراض إعلامية فقط ولا ينبغي أن تحل محل الرأي الطبي المهني.",
        "cancer_signs": "أعراض السرطان",
        "cancer_title": "أعراض السرطان",
        "cancer_content": "تشمل الأعراض الشائعة للسرطان: التعب، وفقدان الوزن غير المبرر، وتغيرات في عادات الأمعاء، والسعال المستمر، والنزيف غير المعتاد أو الإفرازات. يرجى استشارة الطبيب إذا لاحظت أيًا من هذه الأعراض.",
        "contact": "اتصل بنا",
        "contact_title": "معلومات الاتصال",
        "contact_content": "للاستفسارات أو الملاحظات، يرجى الاتصال بنا على: support@example.com",
        "available_langs": ["الإنجليزية", "العربية", "الفرنسية", "الإسبانية"],
        "close": "إغلاق",
        "home": "الرئيسية"
    }
};

// Document Ready Function
document.addEventListener("DOMContentLoaded", function() {
    // Initialize all components
    initializeNavigation();
    initializeAnimations();
    initializeDropdowns();
    initializeLanguageSupport();
});

// Navigation Functions
function initializeNavigation() {
    // Add smooth scrolling for navigation links
    document.querySelectorAll('nav a').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });

    // Add scroll event listener for header
    window.addEventListener('scroll', function() {
        const header = document.querySelector('header');
        if (window.scrollY > 50) {
            header.style.backgroundColor = 'rgba(31, 31, 31, 0.9)';
        } else {
            header.style.backgroundColor = '#1f1f1f';
        }
    });
}

// Animation Functions
function initializeAnimations() {
    const sections = document.querySelectorAll('.section');
    const heroElements = document.querySelectorAll('.hero h1, .hero p, .cta');

    const showSection = (entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    };

    const observer = new IntersectionObserver(showSection, {
        threshold: 0.1
    });

    sections.forEach(section => {
        observer.observe(section);
    });

    setTimeout(() => {
        heroElements.forEach(element => {
            element.style.opacity = '1';
            element.style.transform = 'translateY(0)';
        });
    }, 500);
}

// Dropdown Functions
function initializeDropdowns() {
    const dropdownToggle = document.getElementById('dropdown-toggle');
    const dropdownContent = document.getElementById('dropdown-content');
    const dropdownIcon = document.getElementById('dropdown-icon');

    if (dropdownToggle && dropdownContent && dropdownIcon) {
        dropdownToggle.addEventListener('click', function() {
            dropdownContent.style.display = dropdownContent.style.display === 'block' ? 'none' : 'block';
            dropdownIcon.classList.toggle('rotate');
        });
    }
}

// Language Support Functions
function initializeLanguageSupport() {
    const languageWindow = document.querySelector('.language-window');
    const overlay = document.querySelector('.overlay');
    const closeButton = document.querySelector('.close-button');

    if (languageWindow && overlay && closeButton) {
        // Show language window
        document.querySelector('.change-lang').addEventListener('click', function() {
            addClass(languageWindow, 'show');
            addClass(overlay, 'show');
        });

        // Close language window
        closeButton.addEventListener('click', function() {
            removeClass(languageWindow, 'show');
            removeClass(overlay, 'show');
        });

        // Close on overlay click
        overlay.addEventListener('click', function() {
            removeClass(languageWindow, 'show');
            removeClass(overlay, 'show');
        });
    }
}

// Image Upload Functions
function handleImageUpload(input, previewId) {
    const preview = document.getElementById(previewId);
    const file = input.files[0];
    
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.src = e.target.result;
            preview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
}

// Result Display Functions
function displayResult(result) {
    const resultArea = document.querySelector('.result-area');
    if (resultArea) {
        resultArea.textContent = result ? translations[currentLang].true : translations[currentLang].false;
    }
} 