/* ----------------------------------------------------
   Valuate SaaS Engine - Main Client Javascript
   ---------------------------------------------------- */

document.addEventListener("DOMContentLoaded", () => {
    // Theme selection has been disabled. Light theme is locked.

    // ------------------------------------------------
    // 2. Prediction Form & Async valuation Engine
    // ------------------------------------------------
    const predictForm = document.getElementById("prediction-form");
    if (predictForm) {
        const inputSqft = document.getElementById("input-sqft");
        const inputBedrooms = document.getElementById("input-bedrooms");
        const inputBathrooms = document.getElementById("input-bathrooms");
        
        const btnPredict = document.getElementById("btn-predict");
        const btnReset = document.getElementById("btn-reset");
        const spinner = btnPredict.querySelector(".spinner");
        const btnText = btnPredict.querySelector(".btn-text");
        
        // Result container states
        const resultCard = document.getElementById("valuation-result-card");
        const resultEmpty = document.getElementById("result-empty");
        const resultLoading = document.getElementById("result-loading");
        const resultActive = document.getElementById("result-active");
        
        // Result fields
        const resultCurrency = document.getElementById("result-currency");
        const resultPrice = document.getElementById("result-price");
        const resultSummary = document.getElementById("result-summary");
        const resultConfidence = document.getElementById("result-confidence");
        const resultTimestamp = document.getElementById("result-timestamp");
        const pdfDownloadLink = document.getElementById("pdf-download-link");
        
        // Validator function
        const validateInputs = () => {
            let isValid = true;
            
            // Sqft
            const sqftVal = parseFloat(inputSqft.value);
            const errSqft = document.getElementById("error-sqft");
            if (isNaN(sqftVal) || sqftVal < 100 || sqftVal > 20000) {
                errSqft.style.display = "block";
                inputSqft.classList.add("input-error");
                isValid = false;
            } else {
                errSqft.style.display = "none";
                inputSqft.classList.remove("input-error");
            }
            
            // Bedrooms
            const bedVal = parseInt(inputBedrooms.value);
            const errBedrooms = document.getElementById("error-bedrooms");
            if (isNaN(bedVal) || bedVal < 0 || bedVal > 15) {
                errBedrooms.style.display = "block";
                inputBedrooms.classList.add("input-error");
                isValid = false;
            } else {
                errBedrooms.style.display = "none";
                inputBedrooms.classList.remove("input-error");
            }
            
            // Bathrooms
            const bathVal = parseFloat(inputBathrooms.value);
            const errBathrooms = document.getElementById("error-bathrooms");
            if (isNaN(bathVal) || bathVal < 0.5 || bathVal > 10) {
                errBathrooms.style.display = "block";
                inputBathrooms.classList.add("input-error");
                isValid = false;
            } else {
                errBathrooms.style.display = "none";
                inputBathrooms.classList.remove("input-error");
            }
            
            return isValid;
        };
        
        // Reset state
        const resetResultState = () => {
            resultActive.classList.add("hidden");
            resultLoading.classList.add("hidden");
            resultEmpty.classList.remove("hidden");
        };
        
        btnReset.addEventListener("click", () => {
            predictForm.reset();
            resetResultState();
            
            // Clear errors
            document.querySelectorAll(".error-msg").forEach(el => el.style.display = "none");
            document.querySelectorAll(".form-input").forEach(el => el.classList.remove("input-error"));
        });
        
        // Form submit handler
        predictForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            
            if (!validateInputs()) {
                return;
            }
            
            // Set Loading States
            resultEmpty.classList.add("hidden");
            resultActive.classList.add("hidden");
            resultLoading.classList.remove("hidden");
            
            // Disable button & show spinner
            btnPredict.disabled = true;
            btnReset.disabled = true;
            spinner.style.display = "inline-block";
            btnText.textContent = "Calculating...";
            
            // Read values
            const payload = {
                sqft: parseFloat(inputSqft.value),
                bedrooms: parseInt(inputBedrooms.value),
                bathrooms: parseFloat(inputBathrooms.value)
            };
            
            try {
                // Minimum loading latency of 600ms to allow prediction animation sequence to render beautifully
                const [response] = await Promise.all([
                    fetch("/predict", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify(payload)
                    }),
                    new Promise(resolve => setTimeout(resolve, 600))
                ]);
                
                const data = await response.json();
                
                if (data.success) {
                    // Populate results
                    resultCurrency.textContent = data.currency;
                    // Format price with comma separations
                    resultPrice.textContent = Number(data.price).toLocaleString(undefined, {
                        minimumFractionDigits: 0,
                        maximumFractionDigits: 0
                    });
                    resultSummary.textContent = data.input_summary;
                    resultConfidence.textContent = data.confidence_message;
                    resultTimestamp.textContent = data.timestamp;
                    
                    // PDF Download link mapping
                    pdfDownloadLink.href = `/download-pdf/${data.id}`;
                    
                    // Show active view
                    resultLoading.classList.add("hidden");
                    resultActive.classList.remove("hidden");
                } else {
                    showGlobalToast(data.error || "Prediction engine failed.", "error");
                    resetResultState();
                }
            } catch (err) {
                console.error("Valuation engine connection error:", err);
                showGlobalToast("Network error: Could not reach the valuation server.", "error");
                resetResultState();
            } finally {
                // Restore elements
                btnPredict.disabled = false;
                btnReset.disabled = false;
                spinner.style.display = "none";
                btnText.textContent = "Calculate Estimate";
            }
        });
    }

    // ------------------------------------------------
    // 3. Batch Page Drag and Drop CSV loader
    // ------------------------------------------------
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const fileDetails = document.getElementById("file-details-container");
    
    if (dropZone && fileInput) {
        const fileNameLabel = document.getElementById("selected-file-name");
        const fileSizeLabel = document.getElementById("selected-file-size");
        const removeFileBtn = document.getElementById("btn-remove-file");
        const batchForm = document.getElementById("batch-form");
        const btnSubmitBatch = document.getElementById("btn-submit-batch");
        
        const triggerFileSelection = (file) => {
            if (!file || !file.name.endsWith(".csv")) {
                showGlobalToast("Please upload a valid CSV (.csv) file.", "error");
                return;
            }
            
            // Format size
            const sizeInKb = (file.size / 1024).toFixed(1);
            
            fileNameLabel.textContent = file.name;
            fileSizeLabel.textContent = `${sizeInKb} KB`;
            
            // Reveal details card, hide dropzone contents
            dropZone.classList.add("hidden");
            fileDetails.classList.remove("hidden");
        };
        
        // Dropzone drag-overs
        ["dragenter", "dragover"].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropZone.classList.add("dragover");
            }, false);
        });
        
        ["dragleave", "drop"].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropZone.classList.remove("dragover");
            }, false);
        });
        
        // Handle dropped files
        dropZone.addEventListener("drop", (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length) {
                fileInput.files = files; // Assign files to native file input element
                triggerFileSelection(files[0]);
            }
        });
        
        // Dropzone click browse
        dropZone.addEventListener("click", () => {
            fileInput.click();
        });
        
        // Input change browse
        fileInput.addEventListener("change", (e) => {
            if (fileInput.files.length) {
                triggerFileSelection(fileInput.files[0]);
            }
        });
        
        // Remove selection
        removeFileBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            fileInput.value = ""; // Reset value
            fileDetails.classList.add("hidden");
            dropZone.classList.remove("hidden");
        });
        
        // Batch Form Submit loading spinner
        batchForm.addEventListener("submit", () => {
            btnSubmitBatch.disabled = true;
            const spinnerSubmit = btnSubmitBatch.querySelector(".spinner");
            const textSubmit = btnSubmitBatch.querySelector(".btn-text");
            if (spinnerSubmit && textSubmit) {
                spinnerSubmit.style.display = "inline-block";
                textSubmit.textContent = "Processing CSV matrix...";
            }
        });
    }
    
    // ------------------------------------------------
    // 4. Utility Toast Messages
    // ------------------------------------------------
    const showGlobalToast = (message, type = "success") => {
        let container = document.querySelector(".flash-messages-container");
        if (!container) {
            container = document.createElement("div");
            container.className = "flash-messages-container";
            const mainContent = document.querySelector(".app-main-content");
            if (mainContent) {
                mainContent.insertBefore(container, mainContent.firstChild);
            } else {
                document.body.appendChild(container);
            }
        }
        
        const toast = document.createElement("div");
        toast.className = `flash-message ${type}`;
        toast.role = "alert";
        
        toast.innerHTML = `
            <span class="flash-text">${message}</span>
            <button class="flash-close-btn" aria-label="Dismiss">&times;</button>
        `;
        
        // Add close event
        toast.querySelector(".flash-close-btn").addEventListener("click", () => {
            toast.remove();
        });
        
        container.appendChild(toast);
        
        // Auto remove toast after 5 seconds
        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transition = "opacity 0.5s ease";
            setTimeout(() => toast.remove(), 500);
        }, 5000);
    };
});
