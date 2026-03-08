# Raj Brain - Digital Garden

A personal digital garden built with [Quartz v4](https://quartz.jzhao.xyz/), a static site generator that transforms your Markdown notes into a beautiful, interconnected website.

## How It Works

Quartz v4 is a static site generator that:

1. **Reads Markdown files** from the `content/` folder
2. **Processes them** through a pipeline of transformers (parsing frontmatter, syntax highlighting, etc.)
3. **Filters content** (e.g., removing drafts)
4. **Generates HTML pages** with components like search, graph view, and table of contents
5. **Outputs a static website** to the `public/` folder that can be hosted anywhere

The build process is configured in `quartz.config.ts`, and the page layout is defined in `quartz.layout.ts`.

### Key Files

- **`quartz.config.ts`**: Main configuration file for plugins, theme, fonts, and colors
- **`quartz.layout.ts`**: Defines the layout and components for different page types
- **`content/`**: Where all your Markdown content lives
- **`quartz/styles/`**: SCSS stylesheets for customizing appearance
- **`package.json`**: Dependencies and build scripts

## Where to Write Content

All content should be written in the **`content/`** folder as Markdown files (`.md`).

- **Home page**: `content/index.md` - This is your site's landing page
- **Other pages**: Create any `.md` files in `content/` or subdirectories
- **Recommended**: Use [Obsidian](https://obsidian.md/) as your editor for a better writing experience with features like wikilinks (`[[note-name]]`)

### Content Structure

Each Markdown file can include:

- **Frontmatter** (YAML at the top) for metadata:
  ```yaml
  ---
  title: My Note Title
  tags: [example, note]
  draft: false
  date: 2024-01-01
  ---
  ```

- **Markdown content** with support for:
  - Standard Markdown syntax
  - GitHub Flavored Markdown (tables, footnotes, strikethrough)
  - Obsidian Flavored Markdown (wikilinks `[[note-name]]`, callouts)
  - LaTeX math (via KaTeX)
  - Code blocks with syntax highlighting

## Building and Previewing

To build and preview your site locally:

```bash
npm install
npx quartz build --serve
```

Then open `http://localhost:8081/` in your browser.

To build for production:

```bash
npx quartz build
```

The generated site will be in the `public/` folder.

## Customizing Fonts and Rendering

### Changing Fonts

Fonts are configured in `quartz.config.ts` under the `theme.typography` section:

```typescript
theme: {
  typography: {
    header: "Schibsted Grotesk",  // Font for headings
    body: "Source Sans Pro",       // Font for body text
    code: "IBM Plex Mono",         // Font for code blocks
  },
}
```

You can use any font from [Google Fonts](https://fonts.google.com/). Just specify the font name as a string, or use a more detailed specification:

```typescript
typography: {
  header: {
    name: "Inter",
    weights: [400, 600, 700],
    includeItalic: true,
  },
}
```

### Changing Colors

Colors are also configured in `quartz.config.ts` under `theme.colors`:

```typescript
theme: {
  colors: {
    lightMode: {
      light: "#faf8f8",        // Page background
      darkgray: "#4e4e4e",     // Body text
      dark: "#2b2b2b",         // Header text
      secondary: "#284b63",    // Links
      tertiary: "#84a59d",     // Hover states
      highlight: "rgba(143, 159, 169, 0.15)", // Link backgrounds
    },
    darkMode: {
      // Similar structure for dark mode
    },
  },
}
```

### Custom CSS Styling

For more advanced styling changes, edit `quartz/styles/custom.scss`. This file is imported after the base styles, so you can override any styles here:

```scss
// Example: Change font sizes
body {
  font-size: 18px;
}

// Example: Customize link styles
a {
  text-decoration: underline;
  text-decoration-color: var(--secondary);
}
```

The base styles use CSS variables (like `var(--darkgray)`) that are automatically generated from your color configuration, so you can reference them in your custom styles.

### Layout Breakpoints

To change responsive breakpoints, edit `quartz/styles/variables.scss`:

```scss
$breakpoints: (
  mobile: 800px,   // Below this = mobile layout
  desktop: 1200px, // Above this = desktop layout
);
```

### Changing Page Layout

Edit `quartz.layout.ts` to rearrange components (search bar, table of contents, graph view, etc.) on different parts of the page. See the [layout documentation](https://quartz.jzhao.xyz/layout/) for details.

## Additional Resources

- **Full Documentation**: https://quartz.jzhao.xyz/
- **Configuration Guide**: See `docs/configuration.md`
- **Authoring Content**: See `docs/authoring content.md`
- **Layout Guide**: See `docs/layout.md`

## Getting Help

- [Discord Community](https://discord.gg/cRFFHYye7t)
- [GitHub Issues](https://github.com/jackyzha0/quartz/issues)
