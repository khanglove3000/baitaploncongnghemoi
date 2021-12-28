"""Plotly Dash HTML layout override."""

html_layout = """

<!DOCTYPE html>
<html lang="en">

<head> 
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<!-- Primary Meta Tags -->

<title>
    Flask Volt Dashboard - {% block title %}{% endblock %} | AppSeed
</title>

<meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
<meta name="title" content="Volt - Free Bootstrap 5 Dashboard">
<meta name="author" content="Themesberg">
<meta name="description" content="Volt Pro is a Premium Bootstrap 5 Admin Dashboard featuring over 800 components, 10+ plugins and 20 example pages using Vanilla JS.">
<meta name="keywords" content="bootstrap 5, bootstrap, bootstrap 5 admin dashboard, bootstrap 5 dashboard, bootstrap 5 charts, bootstrap 5 calendar, bootstrap 5 datepicker, bootstrap 5 tables, bootstrap 5 datatable, vanilla js datatable, themesberg, themesberg dashboard, themesberg admin dashboard" />
<link rel="canonical" href="https://appseed.us/admin-dashboards/flask-dashboard-volt">

<!-- Open Graph / Facebook -->
<meta property="og:type" content="website">
<meta property="og:url" content="https://demo.themesberg.com/volt-pro">
<meta property="og:title" content="Volt - Free Bootstrap 5 Dashboard">
<meta property="og:description" content="Volt Pro is a Premium Bootstrap 5 Admin Dashboard featuring over 800 components, 10+ plugins and 20 example pages using Vanilla JS.">
<meta property="og:image" content="https://themesberg.s3.us-east-2.amazonaws.com/public/products/volt-pro-bootstrap-5-dashboard/volt-pro-preview.jpg">

<!-- Twitter -->
<meta property="twitter:card" content="summary_large_image">
<meta property="twitter:url" content="https://demo.themesberg.com/volt-pro">
<meta property="twitter:title" content="Volt - Free Bootstrap 5 Dashboard">
<meta property="twitter:description" content="Volt Pro is a Premium Bootstrap 5 Admin Dashboard featuring over 800 components, 10+ plugins and 20 example pages using Vanilla JS.">
<meta property="twitter:image" content="https://themesberg.s3.us-east-2.amazonaws.com/public/products/volt-pro-bootstrap-5-dashboard/volt-pro-preview.jpg">

<!-- Favicon -->
<link rel="apple-touch-icon" sizes="120x120" href="/static/assets/img/favicon/apple-touch-icon.png">
<link rel="icon" type="image/png" sizes="32x32" href="/static/assets/img/favicon/favicon-32x32.png">
<link rel="icon" type="image/png" sizes="16x16" href="/static/assets/img/favicon/favicon-16x16.png">
<link rel="manifest" href="/static/assets/img/favicon/site.webmanifest">
<link rel="mask-icon" href="/static/assets/img/favicon/safari-pinned-tab.svg" color="#ffffff">
<meta name="msapplication-TileColor" content="#ffffff">
<meta name="theme-color" content="#ffffff">

<link type="text/css" href="/static/assets/vendor/sweetalert2/dist/sweetalert2.min.css" rel="stylesheet">

<link type="text/css" href="/static/assets/vendor/notyf/notyf.min.css" rel="stylesheet">
<link type="text/css" href="/static/assets/css/volt.css" rel="stylesheet">

</head>
        <body class="dash-template">
        


                   <nav class="navbar navbar-dark navbar-theme-primary px-4 col-12 d-lg-none">
            <a class="navbar-brand me-lg-5" href="/">
                <img class="navbar-brand-dark" src="/static/assets/img/brand/light.svg" alt="Volt logo" /> <img class="navbar-brand-light" src="/static/assets/img/brand/dark.svg" alt="Volt logo" />
            </a>
            <div class="d-flex align-items-center">
                <button class="navbar-toggler d-lg-none collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#sidebarMenu" aria-controls="sidebarMenu" aria-expanded="false" aria-label="Toggle navigation">
                  <span class="navbar-toggler-icon"></span>
                </button>
            </div>
        </nav>

        <nav id="sidebarMenu" class="sidebar d-lg-block bg-gray-800 text-white collapse" data-simplebar>
          <div class="sidebar-inner px-4 pt-3">
            
            <ul class="nav flex-column pt-3 pt-md-0">
              <li class="nav-item">
                <a href="/" class="nav-link d-flex align-items-center">
                  <span class="sidebar-icon">
                    <img src="/static/assets/img/brand/light.svg" height="20" width="20" alt="Volt Logo">
                  </span>
                  <span class="mt-1 ms-1 sidebar-text">Chứng Khoáng</span>
                </a>
              </li>
              <li class="nav-item {% if 'dashboard' in segment %} active {% endif %}">
                <a href="/dashapp/" class="nav-link">
                  <span class="sidebar-icon">
                    <svg class="icon icon-xs me-2" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path d="M2 10a8 8 0 018-8v8h8a8 8 0 11-16 0z"></path><path d="M12 2.252A8.014 8.014 0 0117.748 8H12V2.252z"></path></svg>
                  </span> 
                  <span class="sidebar-text">Dashboard</span>
                </a>
              </li>
            </ul>
          </div>
        </nav> 
         <main class="content">

            {%app_entry%}

            <footer class="bg-white rounded shadow p-5 mb-4 mt-4">
    <div class="row">
        <div class="col-12 col-md-4 col-xl-6 mb-4 mb-md-0">
            <p class="mb-0 text-center text-lg-start">
                Phạm Vĩ Khang - 18052441 - Bài tập lớn môn công nghệ mới
            </p>
        </div>
    </div>

         
                {%config%}
                {%scripts%}
                {%renderer%}
    
        </main>
<!-- Core -->
<script src="/static/assets/vendor/@popperjs/core/dist/umd/popper.min.js"></script>
<script src="/static/assets/vendor/bootstrap/dist/js/bootstrap.min.js"></script>

<!-- Vendor JS -->
<script src="/static/assets/vendor/onscreen/dist/on-screen.umd.min.js"></script>

<!-- Slider -->
<script src="/static/assets/vendor/nouislider/distribute/nouislider.min.js"></script>

<!-- Smooth scroll -->
<script src="/static/assets/vendor/smooth-scroll/dist/smooth-scroll.polyfills.min.js"></script>

<!-- Charts -->
<script src="/static/assets/vendor/chartist/dist/chartist.min.js"></script>
<script src="/static/assets/vendor/chartist-plugin-tooltips/dist/chartist-plugin-tooltip.min.js"></script>

<!-- Datepicker -->
<script src="/static/assets/vendor/vanillajs-datepicker/dist/js/datepicker.min.js"></script>

<!-- Sweet Alerts 2 -->
<script src="/static/assets/vendor/sweetalert2/dist/sweetalert2.all.min.js"></script>

<!-- Moment JS -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/moment.js/2.27.0/moment.min.js"></script>

<!-- Vanilla JS Datepicker -->
<script src="/static/assets/vendor/vanillajs-datepicker/dist/js/datepicker.min.js"></script>

<!-- Notyf -->
<script src="/static/assets/vendor/notyf/notyf.min.js"></script>

<!-- Simplebar -->
<script src="/static/assets/vendor/simplebar/dist/simplebar.min.js"></script>

<!-- Github buttons -->
<script async defer src="https://buttons.github.io/buttons.js"></script>

<!-- Volt JS -->
<script src="/static/assets/js/volt.js"></script>

</footer>

         
        </body>
    </html>
"""