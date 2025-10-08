# Batch Processing and Large Run Management

This document explains the enhanced batch processing capabilities designed to handle large product feed evaluations without data loss or timeouts.

## 🚀 New Features

### 1. **Batch Saving**
- **Automatic Progress Saving**: Results are saved in batches to prevent data loss
- **Resume Capability**: Continue from where evaluation stopped
- **Memory Optimization**: Process large datasets without memory issues

### 2. **Health Monitoring**
- **Real-time Health Checks**: Monitor system health during evaluation
- **Rate Limiting**: Automatic API rate limit handling
- **Memory Monitoring**: Track memory usage and prevent crashes
- **Error Tracking**: Comprehensive error logging and analysis

### 3. **Recovery Mechanisms**
- **Automatic Pause**: Pause evaluation when health issues are detected
- **Graceful Recovery**: Resume from the last successful batch
- **Error Resilience**: Continue processing even when individual products fail

## 📁 File Structure

```
product-feed-evaluator/
├── evaluator_v3.py          # Enhanced evaluator with batch saving
├── monitoring.py            # Health monitoring and error handling
├── batch_manager.py         # Batch file management utilities
├── app.py                   # Updated UI with batch controls
└── output/                  # Batch files and logs
    ├── batch_0001_20241201_143022.json
    ├── batch_0002_20241201_143022.json
    ├── progress_20241201_143022.json
    ├── evaluation.log
    └── errors.json
```

## 🔧 Usage

### Basic Batch Processing

```python
from evaluator_v3 import ProductFeedEvaluator

# Initialize with batch saving enabled
evaluator = ProductFeedEvaluator(
    api_key="your-api-key",
    model="gpt-4o-mini",
    locale="en-GB",
    output_dir="output"  # Where to save batches
)

# Process with batch saving
results_df = evaluator.evaluate_products_batch(
    products_df=products_df,
    selected_fields=selected_fields,
    num_questions=12,
    batch_size=20,  # Save every 20 products
    resume=True     # Enable resume capability
)
```

### Resume from Previous Run

```python
# Resume from saved batches
results_df = evaluator.resume_evaluation(products_df)
```

### Health Monitoring

```python
# Get comprehensive status
status = evaluator.get_evaluation_status()
print(f"Health: {status['health_status']['overall_status']}")
print(f"Memory: {status['memory_info']['current_mb']} MB")
print(f"Errors: {status['error_summary']['total_errors']}")
```

## 🛠️ Command Line Tools

### Batch Manager

```bash
# List all batch sessions
python batch_manager.py --list

# Get info about a specific session
python batch_manager.py --info 20241201_143022

# Consolidate a session into CSV
python batch_manager.py --consolidate 20241201_143022

# Clean up old sessions
python batch_manager.py --cleanup 20241201_143022
python batch_manager.py --cleanup-all

# Resume latest session
python batch_manager.py --resume
```

## 📊 Monitoring Dashboard

The Streamlit app now includes a comprehensive monitoring dashboard:

### System Health
- ✅ **Healthy**: All systems operating normally
- ⚠️ **Degraded**: Some issues detected but processing continues
- ❌ **Critical**: Major issues requiring attention

### Memory Usage
- Real-time memory monitoring
- Peak memory tracking
- Automatic recommendations for optimization

### Error Tracking
- Total error count
- Error type breakdown
- Recent error details

### Batch Information
- Latest session timestamp
- Total batches saved
- Progress tracking

## 🔍 Troubleshooting

### Common Issues

#### 1. **High Memory Usage**
**Symptoms**: Slow performance, system freezing
**Solutions**:
- Reduce batch size (try 10-15 instead of 20)
- Enable batch saving to process in smaller chunks
- Monitor memory usage in the dashboard

#### 2. **API Rate Limits**
**Symptoms**: "Rate limit reached" messages
**Solutions**:
- The system automatically handles rate limiting
- Wait times are automatically calculated
- Consider using a higher-tier API plan

#### 3. **Evaluation Timeouts**
**Symptoms**: Process stops with no output
**Solutions**:
- Enable batch saving to save progress
- Use resume functionality to continue
- Check system health in the dashboard

#### 4. **Large Dataset Issues**
**Symptoms**: Out of memory errors, crashes
**Solutions**:
- Process in smaller batches (batch_size=10)
- Enable batch saving
- Monitor memory usage
- Consider processing subsets of data

### Error Recovery

#### Automatic Recovery
The system automatically:
- Saves progress every batch
- Handles API rate limits
- Monitors memory usage
- Pauses when health issues are detected

#### Manual Recovery
1. **Check Status**: Use the monitoring dashboard
2. **Resume**: Click "Resume from previous run"
3. **Clean Up**: Remove old batch files if needed
4. **Restart**: Start fresh if necessary

## 📈 Performance Optimization

### Recommended Settings

#### For Small Datasets (< 100 products)
- Batch size: 20-50
- Batch saving: Optional
- Memory monitoring: Basic

#### For Medium Datasets (100-1000 products)
- Batch size: 15-25
- Batch saving: **Recommended**
- Memory monitoring: **Recommended**

#### For Large Datasets (> 1000 products)
- Batch size: 10-15
- Batch saving: **Required**
- Memory monitoring: **Required**
- Consider processing in subsets

### Memory Optimization Tips

1. **Enable Batch Saving**: Prevents memory accumulation
2. **Monitor Usage**: Watch the memory dashboard
3. **Reduce Batch Size**: Lower batch_size for large datasets
4. **Process Subsets**: Break very large datasets into chunks
5. **Clean Up**: Remove old batch files regularly

## 🔒 Data Safety

### Automatic Backups
- Every batch is automatically saved
- Progress is tracked continuously
- No data loss on system crashes

### Manual Backups
```bash
# Backup all batch files
cp -r output/ backup_$(date +%Y%m%d)/

# Restore from backup
cp -r backup_20241201/output/ ./
```

## 📝 Logging

### Evaluation Logs
- Location: `output/evaluation.log`
- Contains: Progress, errors, health checks
- Format: Timestamped entries

### Error Logs
- Location: `output/errors.json`
- Contains: Detailed error information
- Format: JSON with context and stack traces

### Monitoring Logs
- Real-time health status
- Memory usage tracking
- Rate limiting information

## 🚨 Emergency Procedures

### If Evaluation Stops Unexpectedly

1. **Check Logs**: Look at `output/evaluation.log`
2. **Check Health**: Use the monitoring dashboard
3. **Resume**: Try resuming from the last batch
4. **Clean Restart**: If needed, start fresh with smaller batches

### If Memory Issues Occur

1. **Reduce Batch Size**: Lower to 5-10
2. **Enable Batch Saving**: Ensure it's enabled
3. **Monitor Memory**: Watch the dashboard
4. **Process Subsets**: Break large datasets into chunks

### If API Issues Occur

1. **Check Rate Limits**: Monitor API usage
2. **Wait and Retry**: System handles this automatically
3. **Check API Key**: Ensure it's valid and has sufficient credits
4. **Contact Support**: If issues persist

## 📞 Support

For issues with batch processing:

1. **Check Logs**: Review `output/evaluation.log`
2. **Monitor Health**: Use the dashboard
3. **Try Resume**: Use the resume functionality
4. **Clean Restart**: If all else fails

The enhanced batch processing system is designed to handle large-scale evaluations reliably while providing comprehensive monitoring and recovery capabilities.