#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream> // Required for parsing lines

// Constants for CSV formatting, mirroring pandas defaults
const char DELIMITER = ',';
const char QUOTE_CHAR = '"';
const std::string LINE_TERMINATOR = "\n"; // Pandas often uses \n

// Function to escape quotes within a field
// Replaces " with ""
std::string escape_quotes(const std::string& field) {
    std::string escaped_field;
    for (char c : field) {
        if (c == QUOTE_CHAR) {
            escaped_field += QUOTE_CHAR; // Double the quote
        }
        escaped_field += c;
    }
    return escaped_field;
}

// Function to check if a field needs quoting according to QUOTE_MINIMAL logic
bool needs_quoting(const std::string& field) {
    if (field.find(DELIMITER) != std::string::npos ||
        field.find(QUOTE_CHAR) != std::string::npos ||
        field.find('\n') != std::string::npos || // Check for actual newlines
        field.find('\r') != std::string::npos) { // Check for carriage returns as well
        return true;
    }
    return false;
}

// Function to format a single field for CSV output
std::string format_csv_field(const std::string& field) {
    if (needs_quoting(field)) {
        return QUOTE_CHAR + escape_quotes(field) + QUOTE_CHAR;
    }
    return field;
}

// Function to parse a single CSV line into a vector of fields
// This is a simplified parser; robust CSV parsing can be very complex.
std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string current_field;
    bool in_quotes = false;

    for (size_t i = 0; i < line.length(); ++i) {
        char c = line[i];

        if (in_quotes) {
            if (c == QUOTE_CHAR) {
                // Check for escaped quote ("")
                if (i + 1 < line.length() && line[i + 1] == QUOTE_CHAR) {
                    current_field += QUOTE_CHAR;
                    i++; // Skip next quote
                } else {
                    in_quotes = false; // End of quoted field
                }
            } else {
                current_field += c;
            }
        } else {
            if (c == QUOTE_CHAR) {
                in_quotes = true;
                // If field is not empty and starts with quote, it's part of the quoted content.
                // If field is empty and starts with quote, it's the beginning of a quoted field.
                // This parser assumes quotes are only at the beginning/end of a field if it's quoted,
                // or appear as escaped quotes within.
            } else if (c == DELIMITER) {
                fields.push_back(current_field);
                current_field.clear();
            } else {
                current_field += c;
            }
        }
    }
    fields.push_back(current_field); // Add the last field
    return fields;
}


int main() {
    // Configuration - should match your Python script's settings
    const int DATA_ROWS_TO_COPY = 1000; // Number of data rows (pandas nrows)
    // Ensure this input_file is the same as the one pandas would read
    std::string input_file = "Data/Images_Batched.csv"; // Or your actual input file
    // Output file name to match pandas output (if ROWS is the same)
    std::string output_file = "CopiedFirst" + std::to_string(DATA_ROWS_TO_COPY) + "Rows.csv";

    std::ifstream inFile(input_file);
    std::ofstream outFile(output_file);

    if (!inFile.is_open()) {
        std::cerr << "Error: Could not open input file: " << input_file << std::endl;
        return 1;
    }
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open output file: " << output_file << std::endl;
        inFile.close();
        return 1;
    }

    std::string line;
    int lines_processed_count = 0; // To count header + data rows

    // 1. Read and process the header line
    if (std::getline(inFile, line)) {
        std::vector<std::string> header_fields = parse_csv_line(line);
        for (size_t i = 0; i < header_fields.size(); ++i) {
            outFile << format_csv_field(header_fields[i]);
            if (i < header_fields.size() - 1) {
                outFile << DELIMITER;
            }
        }
        outFile << LINE_TERMINATOR; // Use consistent line terminator
        lines_processed_count++;
    } else {
        std::cerr << "Warning: Input file is empty or header could not be read." << std::endl;
        inFile.close();
        outFile.close();
        return 0; // Or handle as an error
    }

    // 2. Read and process the specified number of data rows
    int data_rows_copied = 0;
    while (data_rows_copied < DATA_ROWS_TO_COPY && std::getline(inFile, line)) {
        // Pandas read_csv with default skip_blank_lines=True might skip empty lines.
        // This simple version doesn't explicitly skip blank lines before counting.
        // If a line is truly blank (empty string), parse_csv_line will return one empty field.
        if (line.empty() && data_rows_copied < DATA_ROWS_TO_COPY) {
             // If pandas skips blank lines, we might need similar logic here.
             // For now, an empty line will be written as such (or as a single empty quoted field if necessary).
             // To truly match pandas, one might need to check if pandas skips this line for nrows counting.
             // This example writes an empty line (which format_csv_field will handle).
        }

        std::vector<std::string> data_fields = parse_csv_line(line);
        for (size_t i = 0; i < data_fields.size(); ++i) {
            outFile << format_csv_field(data_fields[i]);
            if (i < data_fields.size() - 1) {
                outFile << DELIMITER;
            }
        }
        outFile << LINE_TERMINATOR;
        data_rows_copied++;
        lines_processed_count++;
    }

    inFile.close();
    outFile.close();

    std::cout << "Processed " << (lines_processed_count > 0 ? 1 : 0) << " header row and "
              << data_rows_copied << " data rows." << std::endl;
    std::cout << "Output written to " << output_file << std::endl;

    return 0;
}