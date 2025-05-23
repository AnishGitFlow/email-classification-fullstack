import React, { useState, useEffect } from 'react';
import { Send, Mail, Shield, AlertTriangle, CheckCircle, Loader2, BarChart3 } from 'lucide-react';

const SpamDetector = () => {
  const [emailText, setEmailText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [apiStatus, setApiStatus] = useState('checking');

  const sampleTexts = {
    spam1: "CONGRATULATIONS! You have won $1,000,000 in our lottery! Click here immediately to claim your prize before it expires. Send your bank details to claim@lottery-winner.com. Act fast, this offer won't last!",
    spam2: "Amazing weight loss pills! Lose 30 pounds in 30 days guaranteed! No diet, no exercise needed. Special discount 90% OFF today only. Buy now get free shipping worldwide. Limited time offer!",
    ham1: "Hi John, I hope this email finds you well. I wanted to follow up on our meeting yesterday regarding the quarterly budget review. Could you please send me the updated spreadsheet by Friday? Thanks, Sarah",
    ham2: "Meeting Invitation: Project Status Review - Tomorrow 2:00 PM in Conference Room A. Please bring your project updates and we'll discuss the timeline for the next phase. Let me know if you can't attend."
  };

  useEffect(() => {
    checkApiStatus();
  }, []);

  const checkApiStatus = async () => {
    try {
      const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
      const response = await fetch(`${API_URL}/health`);
      if (response.ok) {
        setApiStatus('connected');
      } else {
        setApiStatus('disconnected');
      }
    } catch (error) {
      setApiStatus('disconnected');
    }
  };

  const setSampleText = (key) => {
    setEmailText(sampleTexts[key]);
    setResult(null);
    setError('');
  };

  const simulateAPIResponse = (text) => {
    const spamKeywords = ['win', 'prize', 'money', 'free', 'offer', 'click', 'buy', 'discount', 'limited', 'urgent', 'congratulations', 'guarantee'];
    const hamKeywords = ['meeting', 'project', 'work', 'team', 'schedule', 'update', 'report', 'please', 'thank', 'regards'];
    
    const textLower = text.toLowerCase();
    let spamScore = 0;
    let hamScore = 0;
    
    spamKeywords.forEach(keyword => {
      if (textLower.includes(keyword)) spamScore += 1;
    });
    
    hamKeywords.forEach(keyword => {
      if (textLower.includes(keyword)) hamScore += 1;
    });
    
    const randomFactor = Math.random() * 0.3;
    const baseSpamProb = Math.min(0.9, (spamScore / (spamScore + hamScore + 1)) + randomFactor);
    const spamProbability = isNaN(baseSpamProb) ? 0.5 : baseSpamProb;
    
    return {
      prediction: spamProbability > 0.5 ? 'spam' : 'ham',
      spam_probability: spamProbability,
      ham_probability: 1 - spamProbability,
      confidence: Math.max(spamProbability, 1 - spamProbability)
    };
  };

  const predictSpam = async () => {
    if (!emailText.trim()) {
      setError('Please enter some email content to analyze.');
      return;
    }

    setIsLoading(true);
    setError('');
    setResult(null);

    try {
      const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
      
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: emailText })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const apiResult = await response.json();
      setResult(apiResult);

    } catch (error) {
      console.error('API Error:', error);
      // Fallback to simulation for demo
      setTimeout(() => {
        const mockResult = simulateAPIResponse(emailText);
        setResult(mockResult);
      }, 1500);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
      predictSpam();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 p-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <Shield className="w-12 h-12 text-indigo-600 mr-3" />
            <h1 className="text-4xl font-bold text-gray-800">Email Spam Detection</h1>
          </div>
          <p className="text-lg text-gray-600">AI-powered LSTM model to classify emails as Spam or Ham</p>
          
          {/* API Status */}
          <div className="flex items-center justify-center mt-4">
            <div className={`flex items-center px-3 py-1 rounded-full text-sm ${
              apiStatus === 'connected' ? 'bg-green-100 text-green-800' :
              apiStatus === 'disconnected' ? 'bg-red-100 text-red-800' :
              'bg-yellow-100 text-yellow-800'
            }`}>
              <div className={`w-2 h-2 rounded-full mr-2 ${
                apiStatus === 'connected' ? 'bg-green-500' :
                apiStatus === 'disconnected' ? 'bg-red-500' :
                'bg-yellow-500'
              }`}></div>
              API Status: {apiStatus === 'connected' ? 'Connected' : 
                          apiStatus === 'disconnected' ? 'Using Demo Mode' : 'Checking...'}
            </div>
          </div>
        </div>

        {/* Main Card */}
        <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
          <div className="p-8">
            {/* Input Section */}
            <div className="mb-6">
              <label htmlFor="emailText" className="block text-lg font-semibold text-gray-700 mb-3 flex items-center">
                <Mail className="w-5 h-5 mr-2" />
                Enter Email Content:
              </label>
              <textarea
                id="emailText"
                value={emailText}
                onChange={(e) => setEmailText(e.target.value)}
                onKeyDown={handleKeyPress}
                className="w-full h-48 p-4 border-2 border-gray-200 rounded-xl focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-all duration-200 resize-none bg-gray-50 focus:bg-white"
                placeholder="Paste your email content here...

Example:
Subject: Congratulations! You've won $1,000,000!

Dear winner, you have been selected to receive a cash prize of $1,000,000. Click here to claim your prize now! This offer expires soon."
              />
              <p className="text-sm text-gray-500 mt-2">Tip: Press Ctrl+Enter to analyze</p>
            </div>

            {/* Sample Texts */}
            <div className="mb-6 p-4 bg-gray-50 rounded-xl">
              <h3 className="font-semibold text-gray-700 mb-3">Try these examples:</h3>
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() => setSampleText('spam1')}
                  className="px-4 py-2 bg-red-100 hover:bg-red-200 text-red-700 rounded-full text-sm transition-colors duration-200"
                >
                  🚨 Lottery Scam
                </button>
                <button
                  onClick={() => setSampleText('spam2')}
                  className="px-4 py-2 bg-red-100 hover:bg-red-200 text-red-700 rounded-full text-sm transition-colors duration-200"
                >
                  💊 Fake Medicine
                </button>
                <button
                  onClick={() => setSampleText('ham1')}
                  className="px-4 py-2 bg-green-100 hover:bg-green-200 text-green-700 rounded-full text-sm transition-colors duration-200"
                >
                  📧 Work Email
                </button>
                <button
                  onClick={() => setSampleText('ham2')}
                  className="px-4 py-2 bg-green-100 hover:bg-green-200 text-green-700 rounded-full text-sm transition-colors duration-200"
                >
                  👥 Meeting Invite
                </button>
              </div>
            </div>

            {/* Predict Button */}
            <div className="text-center mb-6">
              <button
                onClick={predictSpam}
                disabled={isLoading || !emailText.trim()}
                className="inline-flex items-center px-8 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white font-semibold rounded-full transition-all duration-200 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
              >
                {isLoading ? (
                  <Loader2 className="w-5 h-5 mr-3 animate-spin" />
                ) : (
                  <Send className="w-5 h-5 mr-3" />
                )}
                {isLoading ? 'Analyzing...' : 'Analyze Email'}
              </button>
            </div>

            {/* Error Message */}
            {error && (
              <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl flex items-center">
                <AlertTriangle className="w-5 h-5 text-red-500 mr-3" />
                <span className="text-red-700">{error}</span>
              </div>
            )}

            {/* Results Section */}
            {result && (
              <div className="bg-gradient-to-r from-gray-50 to-blue-50 rounded-xl p-6 animate-in slide-in-from-bottom duration-500">
                {/* Prediction Badge */}
                <div className="text-center mb-6">
                  <div className={`inline-flex items-center px-6 py-3 rounded-full text-xl font-bold uppercase tracking-wider ${
                    result.prediction === 'spam' 
                      ? 'bg-gradient-to-r from-red-500 to-red-600 text-white' 
                      : 'bg-gradient-to-r from-green-500 to-green-600 text-white'
                  }`}>
                    {result.prediction === 'spam' ? (
                      <AlertTriangle className="w-6 h-6 mr-2" />
                    ) : (
                      <CheckCircle className="w-6 h-6 mr-2" />
                    )}
                    {result.prediction}
                  </div>
                </div>

                {/* Confidence Scores */}
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="bg-white rounded-xl p-6 shadow-sm">
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-sm font-medium text-gray-600 uppercase tracking-wide">Spam Probability</span>
                      <BarChart3 className="w-4 h-4 text-red-500" />
                    </div>
                    <div className="text-3xl font-bold text-red-600 mb-3">
                      {Math.round(result.spam_probability * 100)}%
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-3">
                      <div 
                        className="bg-gradient-to-r from-red-400 to-red-600 h-3 rounded-full transition-all duration-1000 ease-out"
                        style={{ width: `${result.spam_probability * 100}%` }}
                      ></div>
                    </div>
                  </div>

                  <div className="bg-white rounded-xl p-6 shadow-sm">
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-sm font-medium text-gray-600 uppercase tracking-wide">Ham Probability</span>
                      <BarChart3 className="w-4 h-4 text-green-500" />
                    </div>
                    <div className="text-3xl font-bold text-green-600 mb-3">
                      {Math.round(result.ham_probability * 100)}%
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-3">
                      <div 
                        className="bg-gradient-to-r from-green-400 to-green-600 h-3 rounded-full transition-all duration-1000 ease-out"
                        style={{ width: `${result.ham_probability * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>

                {/* Confidence Level */}
                <div className="mt-6 text-center">
                  <div className="text-sm text-gray-600 mb-1">Model Confidence</div>
                  <div className="text-2xl font-bold text-indigo-600">
                    {Math.round(result.confidence * 100)}%
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-8 text-gray-500">
          <p>Powered by TensorFlow LSTM Neural Network</p>
          <p className="text-sm mt-2">
            {apiStatus === 'disconnected' && 'Currently running in demo mode with simulated predictions'}
          </p>
        </div>
      </div>
    </div>
  );
};

export default SpamDetector;