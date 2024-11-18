package main

import (
	"bytes"
	"encoding/xml"
	"fmt"
	"os"
)

// Injects a <style> tag into the <svg> element of an SVG file.
func postProcessGPUsSVG() {
	SVGpath := "static/gpus.svg"

	if err := injectStyleTag(SVGpath); err != nil {
		fmt.Printf("Error injecting style tag into gpus.svg: %v\n", err)
		return
	}
}

// Add <style> tag
func injectStyleTag(SVGpath string) error {
	// Define the style to inject
	styleTag := `
<style type="text/css">
	polygon {
		transition: fill 0.3s linear;
	}
	polygon:hover {
		fill:rgb(37,0,53);
		transition: fill 0s;
	}
</style>`

	// Read the SVG file
	content, err := os.ReadFile(SVGpath)
	if err != nil {
		return fmt.Errorf("error reading SVG file: %w", err)
	}

	// Create a buffer to write the modified content
	var buffer bytes.Buffer
	decoder := xml.NewDecoder(bytes.NewReader(content))

	styleInjected := false

	for {
		// Read each token
		token, err := decoder.Token()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			return fmt.Errorf("error reading SVG tokens: %w", err)
		}

		// Process tokens and write directly to the buffer
		switch t := token.(type) {
		case xml.ProcInst:
			// Write the XML declaration
			buffer.WriteString(fmt.Sprintf("<?%s %s?>\n", t.Target, string(t.Inst)))

		case xml.StartElement:
			// Write the <svg> tag
			buffer.WriteString("<" + t.Name.Local)
			for _, attr := range t.Attr {
				buffer.WriteString(fmt.Sprintf(` %s="%s"`, attr.Name.Local, attr.Value))
			}
			buffer.WriteString(">")

			// Inject the style immediately after the <svg> tag
			if t.Name.Local == "svg" && !styleInjected {
				buffer.WriteString(styleTag)
				styleInjected = true
			}

		case xml.EndElement:
			// Write end elements
			buffer.WriteString(fmt.Sprintf("</%s>", t.Name.Local))

		case xml.CharData:
			// Write character data
			buffer.WriteString(string(t))

		default:
			return fmt.Errorf("unexpected token type: %T", token)
		}
	}

	// Write the modified content back to the file
	if err := os.WriteFile(SVGpath, buffer.Bytes(), 0644); err != nil {
		return fmt.Errorf("error writing modified SVG file: %w", err)
	}

	return nil
}
