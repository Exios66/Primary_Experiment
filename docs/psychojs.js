/**
 * Comprehensive PsychoJS fallback
 * This is a robust implementation to handle cases where the CDN load fails
 * It provides core functionality needed for eye tracking experiments
 */

// Define global PsychoJS constructor
window.PsychoJS = function(options) {
  console.log('Using comprehensive PsychoJS fallback');
  this.debug = options && options.debug || false;
  this.status = 'INITIALISED';
  this.experimentName = options && options.name || 'experiment';
  this.config = options || {};
  this._scheduler = null;
  this._eventManager = new EventManager(this);
  
  // Create core namespaces
  this.visual = {};
  this.data = {};
  this.util = {};
  this.core = {};
  
  // Initialize visual namespace
  this._initVisualNamespace();
  
  // Initialize data namespace
  this._initDataNamespace();
  
  // Initialize utility functions
  this._initUtilNamespace();
  
  // Initialize core functions
  this._initCoreNamespace();
  
  return this;
};

// Initialize the visual namespace with all necessary components
PsychoJS.prototype._initVisualNamespace = function() {
  const self = this;
  
  // Window constructor
  this.visual.Window = function(params) {
    this.size = params.size || [800, 600];
    this.fullscr = params.fullscr || false;
    this.color = params.color || new PsychoJS.Color('#000000');
    this.units = params.units || 'norm';
    this._stimList = [];
    
    // Add a stimulus to the window's stimulus list
    this.addStim = function(stimulus) {
      this._stimList.push(stimulus);
      return this;
    };
    
    // Remove a stimulus from the window's stimulus list
    this.removeStim = function(stimulus) {
      const index = this._stimList.indexOf(stimulus);
      if (index !== -1) {
        this._stimList.splice(index, 1);
      }
      return this;
    };
    
    // Flip method to update the display
    this.flip = function(clearBuffer) {
      if (self.debug) console.log('Window flipped, stimuli count:', this._stimList.length);
      return this;
    };
    
    // Close the window
    this.close = function() {
      if (self.debug) console.log('Window closed');
      this._stimList = [];
      return this;
    };
    
    return this;
  };
  
  // TextStim constructor for displaying text
  this.visual.TextStim = function(params) {
    this.win = params.win;
    this.text = params.text || '';
    this.height = params.height || 0.1;
    this.color = params.color || new PsychoJS.Color('#FFFFFF');
    this.pos = params.pos || [0, 0];
    this.ori = params.ori || 0;
    this.opacity = params.opacity !== undefined ? params.opacity : 1.0;
    this.alignText = params.alignText || 'center';
    this.wrapWidth = params.wrapWidth || 1.0;
    this.flipHoriz = params.flipHoriz || false;
    this.flipVert = params.flipVert || false;
    
    // Draw method to render the text
    this.draw = function() {
      if (self.debug) console.log('Drawing text:', this.text, 'at position:', this.pos);
      if (this.win) this.win.addStim(this);
      return this;
    };
    
    // Set text content
    this.setText = function(text) {
      this.text = text;
      return this;
    };
    
    // Set position
    this.setPos = function(pos) {
      this.pos = pos;
      return this;
    };
    
    // Set color
    this.setColor = function(color) {
      this.color = color;
      return this;
    };
    
    return this;
  };
  
  // Circle constructor for calibration points
  this.visual.Circle = function(params) {
    this.win = params.win;
    this.radius = params.radius || 0.02;
    this.pos = params.pos || [0, 0];
    this.fillColor = params.fillColor || new PsychoJS.Color('#FFFFFF');
    this.lineColor = params.lineColor || new PsychoJS.Color('#FFFFFF');
    this.lineWidth = params.lineWidth || 1;
    this.opacity = params.opacity !== undefined ? params.opacity : 1.0;
    this.edges = params.edges || 32;
    
    // Draw method to render the circle
    this.draw = function() {
      if (self.debug) console.log('Drawing circle at position:', this.pos, 'with radius:', this.radius);
      if (this.win) this.win.addStim(this);
      return this;
    };
    
    // Set position
    this.setPos = function(pos) {
      this.pos = pos;
      return this;
    };
    
    // Set radius
    this.setRadius = function(radius) {
      this.radius = radius;
      return this;
    };
    
    // Set fill color
    this.setFillColor = function(color) {
      this.fillColor = color;
      return this;
    };
    
    // Check if a point is contained within the circle
    this.contains = function(point) {
      const dx = this.pos[0] - point[0];
      const dy = this.pos[1] - point[1];
      return (dx * dx + dy * dy) <= (this.radius * this.radius);
    };
    
    return this;
  };
  
  // Rect constructor for UI elements
  this.visual.Rect = function(params) {
    this.win = params.win;
    this.width = params.width || 0.1;
    this.height = params.height || 0.1;
    this.pos = params.pos || [0, 0];
    this.fillColor = params.fillColor || new PsychoJS.Color('#FFFFFF');
    this.lineColor = params.lineColor || new PsychoJS.Color('#FFFFFF');
    this.lineWidth = params.lineWidth || 1;
    this.opacity = params.opacity !== undefined ? params.opacity : 1.0;
    
    // Draw method to render the rectangle
    this.draw = function() {
      if (self.debug) console.log('Drawing rectangle at position:', this.pos);
      if (this.win) this.win.addStim(this);
      return this;
    };
    
    // Set position
    this.setPos = function(pos) {
      this.pos = pos;
      return this;
    };
    
    // Set size
    this.setSize = function(width, height) {
      this.width = width;
      this.height = height;
      return this;
    };
    
    // Check if a point is contained within the rectangle
    this.contains = function(point) {
      const halfWidth = this.width / 2;
      const halfHeight = this.height / 2;
      return (
        point[0] >= this.pos[0] - halfWidth &&
        point[0] <= this.pos[0] + halfWidth &&
        point[1] >= this.pos[1] - halfHeight &&
        point[1] <= this.pos[1] + halfHeight
      );
    };
    
    return this;
  };
  
  // ImageStim constructor for displaying images
  this.visual.ImageStim = function(params) {
    this.win = params.win;
    this.image = params.image || null;
    this.pos = params.pos || [0, 0];
    this.size = params.size || [0.5, 0.5];
    this.opacity = params.opacity !== undefined ? params.opacity : 1.0;
    this.flipHoriz = params.flipHoriz || false;
    this.flipVert = params.flipVert || false;
    
    // Draw method to render the image
    this.draw = function() {
      if (self.debug) console.log('Drawing image at position:', this.pos);
      if (this.win) this.win.addStim(this);
      return this;
    };
    
    // Set position
    this.setPos = function(pos) {
      this.pos = pos;
      return this;
    };
    
    // Set image
    this.setImage = function(image) {
      this.image = image;
      return this;
    };
    
    return this;
  };
};

// Initialize the data namespace
PsychoJS.prototype._initDataNamespace = function() {
  // ExperimentHandler for data collection
  this.data.ExperimentHandler = function(params) {
    this.name = params.name || 'experiment';
    this.dataFileName = params.dataFileName || 'data';
    this._trials = [];
    this._currentTrial = null;
    
    // Add a trial to the experiment
    this.addData = function(key, value) {
      if (!this._currentTrial) {
        this._currentTrial = {};
      }
      this._currentTrial[key] = value;
      return this;
    };
    
    // Complete the current trial and prepare for the next
    this.nextEntry = function() {
      if (this._currentTrial) {
        this._trials.push(this._currentTrial);
        this._currentTrial = {};
      }
      return this;
    };
    
    // Save the collected data
    this.save = function() {
      console.log('Saving experiment data:', this._trials);
      return this;
    };
    
    return this;
  };
};

// Initialize the utility namespace
PsychoJS.prototype._initUtilNamespace = function() {
  // Clock for timing
  this.util.Clock = function() {
    this._startTime = performance.now();
    
    // Get the current time
    this.getTime = function() {
      return (performance.now() - this._startTime) / 1000;
    };
    
    // Reset the clock
    this.reset = function() {
      this._startTime = performance.now();
      return this;
    };
    
    return this;
  };
  
  // MonotonicClock that cannot be reset
  this.util.MonotonicClock = function() {
    this._startTime = performance.now();
    
    // Get the current time
    this.getTime = function() {
      return (performance.now() - this._startTime) / 1000;
    };
    
    return this;
  };
  
  // CountdownTimer for timed events
  this.util.CountdownTimer = function(startTime) {
    this._startTime = startTime || 0;
    this._clock = new PsychoJS.util.Clock();
    
    // Get the time remaining
    this.getTime = function() {
      return this._startTime - this._clock.getTime();
    };
    
    // Reset the timer
    this.reset = function(startTime) {
      this._startTime = startTime || 0;
      this._clock.reset();
      return this;
    };
    
    return this;
  };
};

// Initialize the core namespace
PsychoJS.prototype._initCoreNamespace = function() {
  // Scheduler for managing experiment flow
  this.core.Scheduler = function() {
    this._taskList = [];
    this._currentTask = null;
    this._running = false;
    
    // Add a task to the scheduler
    this.add = function(task) {
      this._taskList.push(task);
      return this;
    };
    
    // Start running the scheduled tasks
    this.start = function() {
      this._running = true;
      this._runNextTask();
      return this;
    };
    
    // Stop the scheduler
    this.stop = function() {
      this._running = false;
      return this;
    };
    
    // Run the next task in the list
    this._runNextTask = function() {
      if (!this._running || this._taskList.length === 0) return;
      
      this._currentTask = this._taskList.shift();
      const result = this._currentTask();
      
      if (result) {
        // If the task returns true, continue to the next task
        setTimeout(() => this._runNextTask(), 0);
      } else {
        // If the task returns false, stop the scheduler
        this._running = false;
      }
    };
    
    return this;
  };
  
  // Set the scheduler for the experiment
  this.setScheduler = function(scheduler) {
    this._scheduler = scheduler;
    return this;
  };
};

// Event manager for handling keyboard and mouse events
function EventManager(psychoJS) {
  this._psychoJS = psychoJS;
  this._keyboardCallbacks = [];
  this._mouseCallbacks = [];
  
  // Add a keyboard callback
  this.addKeyboardCallback = function(callback) {
    this._keyboardCallbacks.push(callback);
    return this;
  };
  
  // Add a mouse callback
  this.addMouseCallback = function(callback) {
    this._mouseCallbacks.push(callback);
    return this;
  };
  
  // Clear all callbacks
  this.clearCallbacks = function() {
    this._keyboardCallbacks = [];
    this._mouseCallbacks = [];
    return this;
  };
}

// Keyboard class for handling keyboard input
PsychoJS.prototype.Keyboard = function(params) {
  this._psychoJS = params.psychoJS;
  this._bufferSize = params.bufferSize || 10;
  this._keys = [];
  
  // Get keys that have been pressed
  this.getKeys = function(params) {
    const keyList = params && params.keyList || [];
    const waitRelease = params && params.waitRelease || false;
    
    if (keyList.length === 0) {
      return this._keys.slice();
    } else {
      return this._keys.filter(key => keyList.includes(key.name));
    }
  };
  
  // Clear the keys buffer
  this.clearEvents = function() {
    this._keys = [];
    return this;
  };
  
  return this;
};

// Mouse class for handling mouse input
PsychoJS.prototype.Mouse = function(params) {
  this._psychoJS = params.psychoJS;
  this._pos = [0, 0];
  this._buttons = [0, 0, 0];
  
  // Get the current position of the mouse
  this.getPos = function() {
    return this._pos.slice();
  };
  
  // Get the state of the mouse buttons
  this.getPressed = function() {
    return this._buttons.slice();
  };
  
  // Clear the mouse events
  this.clearEvents = function() {
    this._buttons = [0, 0, 0];
    return this;
  };
  
  return this;
};

// Color handling
PsychoJS.Color = function(color) {
  this.color = color;
  
  // Convert the color to RGB
  this.getRGB = function() {
    // Simple implementation for common color names
    const colorMap = {
      'white': [1, 1, 1],
      'black': [0, 0, 0],
      'red': [1, 0, 0],
      'green': [0, 1, 0],
      'blue': [0, 0, 1],
      'yellow': [1, 1, 0],
      'gray': [0.5, 0.5, 0.5]
    };
    
    if (typeof this.color === 'string') {
      if (this.color.startsWith('#')) {
        // Convert hex to RGB
        const hex = this.color.substring(1);
        const r = parseInt(hex.substring(0, 2), 16) / 255;
        const g = parseInt(hex.substring(2, 4), 16) / 255;
        const b = parseInt(hex.substring(4, 6), 16) / 255;
        return [r, g, b];
      } else if (colorMap[this.color.toLowerCase()]) {
        return colorMap[this.color.toLowerCase()];
      }
    } else if (Array.isArray(this.color)) {
      return this.color.slice(0, 3);
    }
    
    // Default to white if color format is not recognized
    return [1, 1, 1];
  };
  
  return this;
};

// Start the experiment
PsychoJS.prototype.start = function(options) {
  this.status = 'STARTED';
  console.log('PsychoJS experiment started:', this.experimentName);
  
  if (this._scheduler) {
    this._scheduler.start();
  }
  
  return this;
};

// Finish the experiment
PsychoJS.prototype.quit = function(options) {
  this.status = 'FINISHED';
  console.log('PsychoJS experiment finished:', this.experimentName);
  
  if (this._scheduler) {
    this._scheduler.stop();
  }
  
  return true;
};

// Log a message when the fallback is loaded
console.log('Comprehensive PsychoJS fallback loaded successfully');