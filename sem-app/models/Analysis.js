const mongoose = require('mongoose');

const analysisSchema = new mongoose.Schema({
    originalFileName: {
        type: String,
        required: true
    },
    uploadDate: {
        type: Date,
        default: Date.now
    },
    excelFilePath: {
        type: String, // Path to the generated Excel file
        required: true
    },
    chartImagePath: {
        type: String, // Path to the generated chart image
        required: true
    },
    sentimentSummary: {
        type: Object, // Store the sentiment summary data
        default: {}
    },
    infraSummary: {
        type: Object, // Store the infrastructure summary data (if applicable)
        default: {}
    },
    modelMetrics: {
        svm: Object,
        rf: Object
    }
});

module.exports = mongoose.model('Analysis', analysisSchema);
