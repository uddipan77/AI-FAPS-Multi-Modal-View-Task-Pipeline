from sklearn.metrics import classification_report
import pandas as pd
import warnings
import wandb


def classification_report_to_excel(y_true, y_pred, classes=None, output_file='classification_report.xlsx', logger_name=None):
    """
    Create an Excel file with the classification report for the given ground truth and predictions.

    """
    # Generate classification report in dictionary format
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Create a DataFrame and transpose it
    report_df = pd.DataFrame(report).transpose()

    # Round the values for better readability
    report_df = report_df.round(2)

    # Class names mapping (replace with your actual class names)
    class_names = dict(enumerate(classes))

    # Create a writer object using xlsxwriter engine
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')

    # Write the DataFrame to Excel
    report_df.to_excel(writer, sheet_name='Report', index=True)

    # Get workbook and sheet for styling
    workbook  = writer.book
    worksheet = writer.sheets['Report']

    # Define a format for the header (bold and background color)
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#D7E4BC',
        'border': 1
    })

    # Apply formatting to header
    for col_num in range(len(report_df.columns)):
        worksheet.write(0, col_num + 1, report_df.columns[col_num], header_format)

    # Save the result
    writer.close()
    
    return report_df
