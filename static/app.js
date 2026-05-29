/* OncoGemma Application Interface Controller */

document.addEventListener("DOMContentLoaded", () => {
    // Initialize Lucide Icons
    lucide.createIcons();

    // Application State Variables
    let selectedSlideFile = null;
    let selectedPdfFile = null;
    let activeProtocol = "H&E";
    let analysisResultData = null;

    // DOM Elements - Input Form
    const analysisForm = document.getElementById("analysis-form");
    const dropzone = document.getElementById("dropzone");
    const dropzoneText = document.getElementById("dropzone-text");
    const slideFileInput = document.getElementById("slide-file");
    const protocolButtons = document.querySelectorAll(".segment-btn");
    const patientNotesInput = document.getElementById("patient-notes");
    const pdfFileInput = document.getElementById("pdf-file");
    const pdfBtnLabel = document.getElementById("pdf-btn-label");
    const submitBtn = document.getElementById("submit-btn");

    // DOM Elements - Pipeline Tracker
    const pipelineBoard = document.getElementById("pipeline-board");
    const stepUpload = document.getElementById("step-upload");
    const uploadStatusText = document.getElementById("upload-status-text");
    const stepExtract = document.getElementById("step-extract");
    const stepVision = document.getElementById("step-vision");
    const stepReport = document.getElementById("step-report");

    // DOM Elements - Results View
    const dashboardView = document.getElementById("dashboard-view");
    const resultsView = document.getElementById("results-view");
    const resultsMetaSubtitle = document.getElementById("results-meta-subtitle");
    const downloadPdfBtn = document.getElementById("download-pdf-btn");
    const resetAnalysisBtn = document.getElementById("reset-analysis-btn");
    const overlayViewer = document.getElementById("overlay-viewer");
    const patchViewer = document.getElementById("patch-viewer");
    const patchCoordinateLabel = document.getElementById("patch-coordinate-label");
    const roiGalleryContainer = document.getElementById("roi-gallery-container");
    const reportViewContainer = document.getElementById("report-view-container");
    const reportPrintContainer = document.getElementById("report-print-container");

    // --- UPLOAD HANDLERS & VALIDATION ---

    // Trigger file input click when clicking dropzone
    dropzone.addEventListener("click", () => {
        slideFileInput.click();
    });

    // Prevent click event bubbling from file input back to dropzone
    slideFileInput.addEventListener("click", (e) => {
        e.stopPropagation();
    });

    // File input selection change
    slideFileInput.addEventListener("change", (e) => {
        handleFileSelect(e.target.files[0]);
    });

    // Drag over effect
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    // Drag leave effect
    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    // File drop event
    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        handleFileSelect(e.dataTransfer.files[0]);
    });

    // Process slide file selection
    function handleFileSelect(file) {
        if (!file) return;
        
        selectedSlideFile = file;
        dropzone.classList.add("has-file");
        dropzoneText.innerText = `Selected Slide: ${file.name} (${formatBytes(file.size)})`;
        
        // Update uploader icon to represent uploaded file
        const iconBox = dropzone.querySelector(".upload-icon-box");
        iconBox.innerHTML = '<i data-lucide="check-circle-2"></i>';
        lucide.createIcons();

        validateForm();
    }

    // PDF attachment handler
    pdfFileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) {
            selectedPdfFile = file;
            pdfBtnLabel.innerText = `Dossier: ${file.name.substring(0, 15)}...`;
            pdfBtnLabel.style.backgroundColor = "rgba(13, 245, 227, 0.08)";
            pdfBtnLabel.style.color = "var(--primary-teal)";
            pdfBtnLabel.style.borderColor = "rgba(13, 245, 227, 0.2)";
        }
    });

    // Toggle exam protocol selection
    protocolButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            protocolButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            activeProtocol = btn.getAttribute("data-exam");
        });
    });

    // Check form completeness to enable submit
    function validateForm() {
        if (selectedSlideFile) {
            submitBtn.removeAttribute("disabled");
            submitBtn.style.animation = "none";
            submitBtn.querySelector("span").innerText = "Initiate AI Diagnostic";
            submitBtn.querySelector("svg").outerHTML = '<i data-lucide="play-circle"></i>';
            lucide.createIcons();
        } else {
            submitBtn.setAttribute("disabled", "true");
        }
    }

    // --- PIPELINE CONTROLLER ---

    // Submit form and run analysis
    analysisForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!selectedSlideFile) return;

        // Reset Pipeline UI
        pipelineBoard.style.display = "block";
        submitBtn.setAttribute("disabled", "true");
        resetPipelineSteps();

        let gcsFileName = null;
        let uploadCompleted = false;

        // Step 1: Upload / Buffer Slide
        setStepState(stepUpload, "active");
        
        try {
            // SVS files can be massive, trigger direct GCS signed URL path if needed
            if (selectedSlideFile.name.toLowerCase().endsWith(".svs")) {
                uploadStatusText.innerText = "Requesting GCS signed upload URL...";
                
                // Get signed URL from backend
                const urlResponse = await fetch("/api/get-upload-url", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        filename: selectedSlideFile.name,
                        contentType: selectedSlideFile.type || "application/octet-stream"
                    })
                });

                if (!urlResponse.ok) throw new Error("Could not acquire GCS upload URL.");
                const urlData = await urlResponse.json();
                
                gcsFileName = urlData.gcsFileName;
                uploadStatusText.innerText = `Buffering WSI slide direct to GCS...`;

                // Perform direct PUT request to GCS signed URL
                const gcsUploadResponse = await fetch(urlData.url, {
                    method: "PUT",
                    headers: {
                        "Content-Type": selectedSlideFile.type || "application/octet-stream"
                    },
                    body: selectedSlideFile
                });

                if (!gcsUploadResponse.ok) throw new Error("WSI direct upload to GCS bucket failed.");
                uploadCompleted = true;
            }

            // Step 1 complete. Start Step 2 extraction simulation
            setStepState(stepUpload, "complete");
            setStepState(stepExtract, "active");

            // Assemble FormData for Backend analysis request
            const formData = new FormData();
            formData.append("examType", activeProtocol);
            formData.append("patientReport", patientNotesInput.value);
            if (selectedPdfFile) {
                formData.append("patientPdf", selectedPdfFile);
            }
            formData.append("originalFileName", selectedSlideFile.name);

            if (gcsFileName && uploadCompleted) {
                formData.append("gcsFileName", gcsFileName);
            } else {
                // If standard image or GCS skipped, upload file directly inside multi-part Form Data
                formData.append("image", selectedSlideFile);
            }

            // Start visual timer simulation to keep pipeline tracker alive
            const pipelineTimer = runPipelineSimulation();

            // Fire main analyze request to FastAPI
            const analyzeResponse = await fetch("/api/analyze", {
                method: "POST",
                body: formData
            });

            clearInterval(pipelineTimer);

            if (!analyzeResponse.ok) {
                const errorData = await analyzeResponse.json();
                throw new Error(errorData.detail || errorData.error || "Analysis pipeline execution failed.");
            }

            const data = await analyzeResponse.json();
            if (!data.success) {
                throw new Error(data.error || "Analysis failed.");
            }

            // Complete remaining steps instantly
            setStepState(stepExtract, "complete");
            setStepState(stepVision, "complete");
            setStepState(stepReport, "complete");

            // Store result data & render
            analysisResultData = data;
            
            setTimeout(() => {
                renderResultsView();
            }, 600);

        } catch (error) {
            console.error("Analysis error:", error);
            alert(`Diagnostic Pipeline Interrupted:\n${error.message}`);
            
            // Restore Form UI
            pipelineBoard.style.display = "none";
            validateForm();
        }
    });

    // Simulate progress tracker state changes
    function runPipelineSimulation() {
        let elapsed = 0;
        return setInterval(() => {
            elapsed += 1;
            if (elapsed === 4) {
                setStepState(stepExtract, "complete");
                setStepState(stepVision, "active");
            } else if (elapsed === 10) {
                setStepState(stepVision, "complete");
                setStepState(stepReport, "active");
            }
        }, 1000);
    }

    // Set styling state on pipeline step cards
    function setStepState(element, state) {
        element.classList.remove("active", "complete");
        if (state === "active") {
            element.classList.add("active");
        } else if (state === "complete") {
            element.classList.add("complete");
            element.querySelector(".step-indicator").innerHTML = '<i data-lucide="check" style="width: 1.1rem; height: 1.1rem;"></i>';
            lucide.createIcons();
        }
    }

    function resetPipelineSteps() {
        [stepUpload, stepExtract, stepVision, stepReport].forEach((step, idx) => {
            step.classList.remove("active", "complete");
            step.querySelector(".step-indicator").innerText = idx + 1;
        });
        uploadStatusText.innerText = "Buffering slide file onto GCS temporary nodes...";
        lucide.createIcons();
    }

    // --- RESULTS VIEWER & INTERACTIONS ---

    function renderResultsView() {
        if (!analysisResultData) return;

        // Swap views
        dashboardView.style.display = "none";
        resultsView.style.display = "block";

        resultsMetaSubtitle.innerText = `${activeProtocol} staining evaluation completed.`;

        // Load Thumbnail & Overlay
        overlayViewer.src = analysisResultData.overlayUrl || "/static/data/slide_thumbnail.png";
        
        // Render markdown report using Marked library
        const rawMarkdown = analysisResultData.report || "No report generated.";
        reportViewContainer.innerHTML = marked.parse(rawMarkdown);

        // Build patches visual ROI gallery
        roiGalleryContainer.innerHTML = "";
        const patches = analysisResultData.patches || [];

        if (patches.length > 0) {
            // Select first patch as active by default
            selectPatch(patches[0]);

            patches.forEach((patch, idx) => {
                const thumb = document.createElement("div");
                thumb.className = `roi-gallery-thumb ${idx === 0 ? 'active' : ''}`;
                thumb.innerHTML = `<img src="${patch.url}" alt="Crop ${idx}">`;
                
                thumb.addEventListener("click", () => {
                    document.querySelectorAll(".roi-gallery-thumb").forEach(t => t.classList.remove("active"));
                    thumb.classList.add("active");
                    selectPatch(patch);
                });
                
                roiGalleryContainer.appendChild(thumb);
            });
        } else {
            // Standard image uploaded (no WSI crop list)
            patchViewer.src = analysisResultData.thumbnailUrl;
            patchCoordinateLabel.innerText = "Standard image view (No coordinate offsets)";
        }

        // Setup printable cloned report container
        reportPrintContainer.innerHTML = marked.parse(rawMarkdown);

        // Append selected WSI patches to the printable dossier if present
        if (patches.length > 0) {
            let patchesHtml = `
                <hr style="margin-top: 30px; border-top: 2px solid #000;"/>
                <h2 style="font-size: 14pt; font-weight: bold; margin-top: 20px; color: #000; font-family: sans-serif;">SELECTED PATHOLOGY HOTSPOTS (TISSUE ROIs)</h2>
                <p style="font-size: 10pt; color: #555; margin-bottom: 15px;">The following ${patches.length} high-power field (HPF) patches were automatically selected based on tissue density (>70% cellularity) for morphological scoring.</p>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 15px;">
            `;
            
            patches.forEach((patch, idx) => {
                patchesHtml += `
                    <div style="border: 1px solid #ccc; padding: 10px; text-align: center; background: #fafafa; border-radius: 5px; box-sizing: border-box; page-break-inside: avoid;">
                        <img src="${patch.url}" style="width: 100%; max-width: 250px; height: auto; display: block; margin: 0 auto 8px; border-radius: 3px; border: 1px solid #ddd;"/>
                        <div style="font-size: 8.5pt; font-family: monospace; color: #333; font-weight: bold;">
                            ROI #${idx + 1} &mdash; Field ID: ${patch.filename}
                        </div>
                        <div style="font-size: 7.5pt; font-family: monospace; color: #666; margin-top: 2px;">
                            Coordinate: [x=${patch.x}, y=${patch.y}] | Region: ${patch.region}
                        </div>
                    </div>
                `;
            });
            
            patchesHtml += `</div>`;
            reportPrintContainer.innerHTML += patchesHtml;
        } else if (analysisResultData.thumbnailUrl) {
            let imageHtml = `
                <hr style="margin-top: 30px; border-top: 2px solid #000;"/>
                <h2 style="font-size: 14pt; font-weight: bold; margin-top: 20px; color: #000; font-family: sans-serif;">ANALYZED CLINICAL SPECIMEN</h2>
                <div style="text-align: center; margin-top: 15px; page-break-inside: avoid;">
                    <img src="${analysisResultData.thumbnailUrl}" style="max-width: 100%; max-height: 400px; border: 1px solid #ccc; padding: 5px; background: #fff;"/>
                </div>
            `;
            reportPrintContainer.innerHTML += imageHtml;
        }
    }

    function selectPatch(patch) {
        patchViewer.src = patch.url;
        patchCoordinateLabel.innerText = `Field ID: ${patch.filename} | Coordinate: [x=${patch.x}, y=${patch.y}]`;
    }

    // Reset back to upload form dashboard
    resetAnalysisBtn.addEventListener("click", () => {
        analysisResultData = null;
        selectedSlideFile = null;
        selectedPdfFile = null;
        
        // Reset Inputs
        slideFileInput.value = "";
        pdfFileInput.value = "";
        patientNotesInput.value = "";
        
        pdfBtnLabel.innerText = "Browse PDF";
        pdfBtnLabel.style = "";
        
        dropzone.classList.remove("has-file");
        dropzoneText.innerText = "Click or drag WSI slide here";
        dropzone.querySelector(".upload-icon-box").innerHTML = '<i data-lucide="upload-cloud"></i>';
        
        // Swap Views
        resultsView.style.display = "none";
        pipelineBoard.style.display = "none";
        dashboardView.style.display = "block";
        
        validateForm();
        lucide.createIcons();
    });

    // --- PDF DOSSIER DOWNLOAD ---
    downloadPdfBtn.addEventListener("click", () => {
        if (!analysisResultData) return;

        const opt = {
            margin: 15,
            filename: `OncoGemma_Synoptic_Report_${Date.now()}.pdf`,
            image: { type: 'jpeg', quality: 1.0 },
            html2canvas: { scale: 2, useCORS: true, logging: false },
            jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
        };

        // Render PDF directly using html2pdf library targeting our white-paper styled printable print container
        html2pdf().set(opt).from(reportPrintContainer).save();
    });

    // Utilities
    function formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }
});
