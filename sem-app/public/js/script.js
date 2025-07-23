document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const loadingDiv = document.getElementById('loading');
    const errorMessageDiv = document.getElementById('error-message');

    if (uploadForm) { // Only run this on the home page where the form exists
        uploadForm.addEventListener('submit', async (event) => {
            event.preventDefault();

            loadingDiv.classList.remove('hidden');
            errorMessageDiv.classList.add('hidden');

            const formData = new FormData(uploadForm);

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData,
                });

                loadingDiv.classList.add('hidden');

                if (response.ok) {
                    const data = await response.json();
                    if (data.redirectUrl) {
                        window.location.href = data.redirectUrl; // Redirect to results page
                    } else {
                        errorMessageDiv.textContent = 'Analysis complete, but no redirect URL provided.';
                        errorMessageDiv.classList.remove('hidden');
                    }
                } else {
                    const errorData = await response.json();
                    errorMessageDiv.textContent = `Error: ${errorData.error || 'Something went wrong.'} ${errorData.details || ''}`;
                    errorMessageDiv.classList.remove('hidden');
                }
            } catch (error) {
                console.error('Fetch error:', error);
                loadingDiv.classList.add('hidden');
                errorMessageDiv.textContent = 'An unexpected error occurred. Please try again.';
                errorMessageDiv.classList.remove('hidden');
            }
        });
    }

    // You could add Chart.js rendering logic here if you decide to send raw data
    // instead of an image, but for now, we're relying on the Python-generated image.
});
