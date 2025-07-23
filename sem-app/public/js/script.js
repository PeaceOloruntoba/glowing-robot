document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const loadingDiv = document.getElementById('loading');
    const loadingMessage = document.getElementById('loading-message');
    const uploadButton = document.querySelector('#uploadForm button[type="submit"]');
    const errorMessageDiv = document.getElementById('error-message');

    if (uploadForm) { // Only run this on the home page where the form exists
        uploadForm.addEventListener('submit', async (event) => {
            event.preventDefault();

            loadingDiv.classList.remove('hidden');
            loadingMessage.textContent = 'Starting analysis...'; // Initial message
            uploadButton.disabled = true; // Disable the button
            errorMessageDiv.classList.add('hidden');

            const formData = new FormData(uploadForm);

            try {
                // For now, we update the loading message in a simple progression.
                // For true real-time streaming of Python logs, WebSockets/SSE would be required.
                loadingMessage.textContent = 'Uploading file and preparing for analysis...';

                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData,
                });

                loadingDiv.classList.add('hidden'); // Hide loading after response received
                uploadButton.disabled = false; // Re-enable the button

                if (response.ok) {
                    const data = await response.json();
                    if (data.redirectUrl) {
                        window.location.href = data.redirectUrl; // Redirect to results page
                    } else {
                        errorMessageDiv.textContent = 'Analysis complete, but no redirect URL provided by the server.';
                        errorMessageDiv.classList.remove('hidden');
                    }
                } else {
                    const errorData = await response.json();
                    let displayMessage = `Error: ${errorData.error || 'Something went wrong.'}`;
                    if (errorData.details) {
                        const details = String(errorData.details); // Ensure it's a string

                        // Attempt to find specific Python error messages
                        const lookupErrorMatch = details.match(/LookupError: \*{5,}\s*Resource\s*(\S+)\s*not found\./);
                        const criticalNLTKErrorMatch = details.match(/CRITICAL ERROR: Failed to download NLTK resource '(\S+)'\. Please download manually\./);

                        if (lookupErrorMatch) {
                            displayMessage = `Error: Required NLTK resource '${lookupErrorMatch[1]}' not found. Please ensure it's downloaded in your Python environment.`;
                        } else if (criticalNLTKErrorMatch) {
                            displayMessage = `Error: NLTK resource '${criticalNLTKErrorMatch[1]}' failed to download automatically. Please run 'import nltk; nltk.download(\'${criticalNLTKErrorMatch[1]}\')' in your Python environment.`;
                        } else {
                            // Fallback to showing a snippet of the error if specific patterns aren't found
                            displayMessage += ` Details: ${details.substring(0, Math.min(details.length, 300))}...`; // Show first 300 chars or less
                        }
                    }
                    errorMessageDiv.textContent = displayMessage;
                    errorMessageDiv.classList.remove('hidden');
                }
            } catch (error) {
                console.error('Fetch error:', error);
                loadingDiv.classList.add('hidden');
                uploadButton.disabled = false; // Re-enable on network error too
                errorMessageDiv.textContent = 'An unexpected network error occurred. Please check your connection and try again.';
                errorMessageDiv.classList.remove('hidden');
            }
        });
    }

    // No Chart.js rendering client-side as we're using Python-generated images.
});
