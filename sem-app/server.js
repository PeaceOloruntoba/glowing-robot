require('dotenv').config();
const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const exphbs = require('express-handlebars');
const mongoose = require('mongoose'); // Import Mongoose
const Analysis = require('./models/Analysis'); // Import the Analysis model

const app = express();
const PORT = process.env.PORT || 3000;
const MONGODB_URI = process.env.MONGODB_URI;

// Connect to MongoDB
mongoose.connect(MONGODB_URI)
    .then(() => console.log('MongoDB connected successfully!'))
    .catch(err => console.error('MongoDB connection error:', err));

// Set up Handlebars as the templating engine
app.engine('.hbs', exphbs.engine({
    extname: '.hbs',
    defaultLayout: 'main',
    layoutsDir: path.join(__dirname, 'views/layouts'),
    partialsDir: path.join(__dirname, 'views/partials'),
    helpers: {
        // Helper to format dates for display
        formatDate: (date) => {
            return new Date(date).toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'long',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        },
        // Helper for conditional checks in Handlebars (e.g., if infraSummary exists)
        ifExists: (value, options) => {
            if (value && Object.keys(value).length > 0) {
                return options.fn(this);
            }
            return options.inverse(this);
        }
    }
}));
app.set('view engine', '.hbs');
app.set('views', path.join(__dirname, 'views'));

// Create necessary directories if they don't exist
const uploadsDir = path.join(__dirname, 'uploads');
const analysisResultsBaseDir = path.join(__dirname, 'analysis_results');
const analysisFilesDir = path.join(analysisResultsBaseDir, 'analysis_files'); // For Excel files
const analysisImagesDir = path.join(analysisResultsBaseDir, 'images'); // For chart images

fs.existsSync(uploadsDir) || fs.mkdirSync(uploadsDir);
fs.existsSync(analysisResultsBaseDir) || fs.mkdirSync(analysisResultsBaseDir);
fs.existsSync(analysisFilesDir) || fs.mkdirSync(analysisFilesDir, { recursive: true });
fs.existsSync(analysisImagesDir) || fs.mkdirSync(analysisImagesDir, { recursive: true });

// Configure Multer for file uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, uploadsDir);
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + '-' + file.originalname);
    }
});
const upload = multer({ storage: storage });

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));
app.use('/analysis_results', express.static(analysisResultsBaseDir)); // Serve both images and excel files

// --- Routes ---

// Home page
app.get('/', (req, res) => {
    res.render('home', {
        title: 'Sentiment Analysis Project - Upload'
    });
});

// History page
app.get('/history', async (req, res) => {
    try {
        const analyses = await Analysis.find().sort({ uploadDate: -1 });
        res.render('history', {
            title: 'Sentiment Analysis History',
            analyses: analyses
        });
    } catch (err) {
        console.error('Error fetching analysis history:', err);
        res.status(500).render('error', { message: 'Failed to load analysis history.' });
    }
});

// Analysis results page (for both new and historical results)
app.get('/analysis/:id', async (req, res) => {
    try {
        const analysis = await Analysis.findById(req.params.id);
        if (!analysis) {
            return res.status(404).render('error', { message: 'Analysis not found.' });
        }
        res.render('results', {
            title: `Analysis Results for ${analysis.originalFileName}`,
            analysis: analysis,
            // Pass stringified JSON for client-side processing if needed
            sentimentSummaryJson: JSON.stringify(analysis.sentimentSummary),
            infraSummaryJson: JSON.stringify(analysis.infraSummary)
        });
    } catch (err) {
        console.error('Error fetching analysis details:', err);
        res.status(500).render('error', { message: 'Failed to load analysis details.' });
    }
});


// File upload and analysis POST route
app.post('/upload', upload.single('dataset'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded.' });
    }

    const filePath = req.file.path;
    const originalFileName = req.file.originalname;
    const isCsv = req.file.mimetype === 'text/csv';

    const pythonExecutable = 'python'; // Or 'python3'
    const pythonScriptPath = path.join(__dirname, 'python_script', 'sentiment_analysis.py');

    // Pass output directories to Python script
    const args = [
        pythonScriptPath,
        filePath,
        isCsv ? 'csv' : 'other',
        analysisFilesDir, // Pass Excel output directory
        analysisImagesDir // Pass Chart output directory
    ];

    const pythonProcess = spawn(pythonExecutable, args);

    let pythonOutput = '';
    let pythonError = '';

    pythonProcess.stdout.on('data', (data) => {
        pythonOutput += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        pythonError += data.toString();
        console.error(`Python stderr: ${data}`);
    });

    pythonProcess.on('close', async (code) => {
        // Clean up the uploaded file
        fs.unlink(filePath, (err) => {
            if (err) console.error('Error deleting uploaded file:', err);
        });

        if (code !== 0) {
            console.error(`Python script exited with code ${code}`);
            return res.status(500).json({ error: 'Analysis failed', details: pythonError });
        }

        try {
            // Python script should print paths/data in a parseable format (e.g., JSON)
            // For now, we'll assume it saves files with specific names and we'll parse stdout for the unique excel filename.
            const outputLines = pythonOutput.split('\n');
            const excelFilenameLine = outputLines.find(line => line.startsWith('Output Excel file:'));
            const chartFilenameLine = outputLines.find(line => line.startsWith('Sentiment distribution chart:'));

            let excelFilename = null;
            let chartFilename = null;
            let sentimentSummaryFromPython = {};
            let infraSummaryFromPython = {};
            let modelMetricsFromPython = {};

            if (excelFilenameLine) {
                excelFilename = excelFilenameLine.split(': ')[1].trim();
            }
            if (chartFilenameLine) {
                chartFilename = chartFilenameLine.split(': ')[1].trim();
            }

            // Attempt to parse any JSON output from Python (e.g., sentiment summary)
            try {
                const jsonOutputStart = pythonOutput.indexOf('{');
                const jsonOutputEnd = pythonOutput.lastIndexOf('}');
                if (jsonOutputStart !== -1 && jsonOutputEnd !== -1) {
                    const jsonString = pythonOutput.substring(jsonOutputStart, jsonOutputEnd + 1);
                    const parsedPythonData = JSON.parse(jsonString);
                    sentimentSummaryFromPython = parsedPythonData.sentiment_summary || {};
                    infraSummaryFromPython = parsedPythonData.infra_summary || {};
                    modelMetricsFromPython = parsedPythonData.model_metrics || {};
                }
            } catch (jsonParseError) {
                console.warn("Could not parse JSON from Python stdout:", jsonParseError);
                // Continue without JSON data if parsing fails
            }

            if (!excelFilename || !chartFilename) {
                return res.status(500).json({ error: 'Analysis completed but output files not clearly identified from Python script output.', pythonOutput });
            }

            const excelRelativePath = path.join('analysis_files', excelFilename);
            const chartRelativePath = path.join('images', chartFilename);

            // Save analysis details to MongoDB
            const newAnalysis = new Analysis({
                originalFileName: originalFileName,
                excelFilePath: excelRelativePath,
                chartImagePath: chartRelativePath,
                sentimentSummary: sentimentSummaryFromPython,
                infraSummary: infraSummaryFromPython,
                modelMetrics: modelMetricsFromPython
            });

            const savedAnalysis = await newAnalysis.save();
            console.log('Analysis saved to DB:', savedAnalysis._id);

            // Redirect to the results page for the newly saved analysis
            res.json({
                message: 'Analysis complete!',
                redirectUrl: `/analysis/${savedAnalysis._id}`
            });

        } catch (dbSaveError) {
            console.error('Error saving analysis to database:', dbSaveError);
            res.status(500).json({ error: 'Analysis complete but failed to save results to database.', details: dbSaveError.message });
        }
    });
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
