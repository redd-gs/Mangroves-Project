# Google Earth Engine Credentials

## How to Obtain

1. Go to https://code.earthengine.google.com/
2. Sign in with your Google account
3. Register for Earth Engine access (if not already done)
4. Create or access your GEE project
5. Copy your project ID

## Setup

Create a file named `gee_project_key.txt` in this directory:

```
your-project-id-here
```

**Important:** This file is protected by .gitignore and will never be committed to version control.

## Usage

The pipeline will automatically read this file when accessing Google Earth Engine data.

⚠️ **Security Warning:** Never share this file or commit it to git repositories.
